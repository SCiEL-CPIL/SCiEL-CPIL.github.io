"""Broadband antireflection coating design.

Strategy (fresh design):
  1. Parametric continuous effective-index profile n(z) = n_air + (n_sub-n_air)*s(z),
     where s(z) is a smooth monotone shape (cubic polynomial / sqrt / log).
  2. Discretize into a fixed number of discrete layers, each assigned material L or H
     (whichever index is closer to the local target) and a thickness equal to the
     optical path length divided by that material's index.
  3. Coarse search over profile-shape parameters + layer count + material-pattern
     swaps to get a good discrete design.
  4. Local Nelder-Mead polish on the discrete thicknesses (with a layer-count /
     material-pattern escape) to refine.
  5. Deadline guard: track elapsed wall-clock time and return the best design found
     with a safety margin before the 150 s limit.
"""

import time
import math

import numpy as np

# Wall-clock deadline (safety margin under the 150 s kill).
_DEADLINE = 140.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _reflectance_spectrum(materials, thicknesses, problem, wavelengths):
    """Exact normal-incidence reflectance (0..1) via characteristic-matrix method.

    materials: list of 'L'/'H'
    thicknesses: list of physical thicknesses in nm (same length)
    wavelengths: 1-D array of wavelengths in nm
    Returns array of reflectance in [0,1].
    """
    n_map = problem["materials"]
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]

    n_layers = [n_map[m] for m in materials]
    k = len(materials)

    # w = 2*pi*lambda
    w = 2.0 * np.pi * wavelengths  # shape (N,)
    # phase per layer: phi_j = (2*pi/lambda) * n_j * d_j = w * n_j * d_j
    # = 2*pi*n_j*d_j/lambda ; since w=2*pi*lambda, phi_j = n_j*d_j*lambda/lambda... 
    # Actually phase = (2*pi/lambda)*n*d. With w=2*pi*lambda, phase = n*d*lambda? No.
    # Let's do it directly: phase_j(λ) = 2*pi*n_j*d_j/λ.

    # We'll compute for each wavelength.
    N = len(wavelengths)
    R = np.empty(N)
    for i, lam in enumerate(wavelengths):
        # Build the characteristic matrix product M = M1 M2 ... Mk
        # M_j = [[cos(phi_j), i*sin(phi_j)/n_j],
        #        [i*n_j*sin(phi_j), cos(phi_j)]]
        # Work in complex.
        M = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                      [0.0 + 0.0j, 1.0 + 0.0j]])
        for j in range(k):
            n_j = n_layers[j]
            d_j = thicknesses[j]
            phi = 2.0 * np.pi * n_j * d_j / lam
            c = math.cos(phi)
            s = math.sin(phi)
            Mj = np.array([[c + 0.0j, 1j * s / n_j],
                           [1j * n_j * s, c + 0.0j]])
            M = M @ Mj
        # admittance B,C: [B;C] = M @ [1; 1/n_sub]
        B = M[0, 0] * 1.0 + M[0, 1] * (1.0 / n_sub)
        C = M[1, 0] * 1.0 + M[1, 1] * (1.0 / n_sub)
        # reflectance amplitude r = (n_amb*B - C)/(n_amb*B + C)
        r = (n_amb * B - C) / (n_amb * B + C)
        R[i] = (r * np.conj(r)).real
    return np.clip(R, 0.0, 1.0)


def _score(materials, thicknesses, problem, wavelengths):
    """Mean reflectance in percent over the band (minimize)."""
    R = _reflectance_spectrum(materials, thicknesses, problem, wavelengths)
    return 100.0 * float(np.mean(R))


def _profile_shape(z, shape, a):
    """Monotone shape s(z) in [0,1] on z in [0,1].

    shape: 'poly' (cubic), 'sqrt', 'log', 'smoothstep'
    a: shape parameter in [0,1].
    """
    z = np.asarray(z)
    if shape == "poly":
        # cubic s = z*(a + (1-2a)*z + a*z^2)? ensure s(0)=0,s(1)=1
        # Use s = z^p with p varying? Keep monotone via blend of z and z^2.
        # s = (1-a)*z + a*z^2  (monotone, s(0)=0,s(1)=1)
        return (1.0 - a) * z + a * z * z
    if shape == "sqrt":
        # blend between z and sqrt(z)
        return (1.0 - a) * z + a * np.sqrt(z)
    if shape == "log":
        # blend between z and log shape: use (z^(1-a))
        return z ** (1.0 - a)
    if shape == "smoothstep":
        # s = z^2*(3-2z) blended with z
        ss = z * z * (3.0 - 2.0 * z)
        return (1.0 - a) * z + a * ss
    raise ValueError("unknown shape %r" % (shape,))


def _profile_to_layers(shape, a, n_layers, n_air, n_sub, max_total_nm, max_layer_nm):
    """Discretize a continuous profile into discrete L/H layers.

    Returns (materials, thicknesses) or None if infeasible.
    """
    # total optical path to distribute; use a fraction of the budget so we leave
    # headroom and respect per-layer cap.
    # Target total physical thickness: use up to max_total_nm but ensure each layer
    # <= max_layer_nm. For n_layers layers, max per layer = max_total_nm/n_layers.
    per_layer_max = min(max_layer_nm, max_total_nm / float(n_layers))

    # Build the profile on layer-centre positions.
    z = (np.arange(n_layers) + 0.5) / float(n_layers)  # in (0,1)
    s = _profile_shape(z, shape, a)
    s = np.clip(s, 0.0, 1.0)
    n_target = n_air + (n_sub - n_air) * s

    materials = []
    thicknesses = []
    nL = 1.38
    nH = 2.35
    for nt in n_target:
        # choose material whose index is closer
        if abs(nt - nL) <= abs(nt - nH):
            m = "L"
            nm = nL
        else:
            m = "H"
            nm = nH
        materials.append(m)

    # Assign thicknesses: we want the optical path of each layer to match the
    # target index * segment length. Distribute total physical thickness so that
    # the optical path matches n_target * (segment). Total optical path budget:
    # use a chosen total physical thickness T (variable) so mean optical index
    # * T approximates the integral of n_target.
    # We'll set T = max_total_nm * t_frac, then scale per-layer thickness so the
    # optical path of layer j = n_target_j * (T/n_layers) (i.e., physical thickness
    # d_j = n_target_j/nm_j * (T/n_layers)).
    # This keeps total physical thickness = T * mean(n_target/nm) which may exceed
    # T; we then clamp to per_layer_max and total.
    return materials, per_layer_max, n_target, nL, nH


def _make_design(shape, a, n_layers, problem):
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    materials, per_layer_max, n_target, nL, nH = _profile_to_layers(
        shape, a, n_layers, n_air, n_sub, max_total_nm, max_layer_nm)

    # Choose total physical thickness T so that optical path matches target.
    # We want sum_j nm_j * d_j = sum_j n_target_j * (T0/n_layers) where T0 is a
    # reference total. Simpler: set d_j = n_target_j / nm_j * (T/n_layers) and pick
    # T = max_total_nm (use full budget), then clamp.
    T = max_total_nm
    seg = T / float(n_layers)
    thicknesses = []
    for j, nt in enumerate(n_target):
        nm = nL if materials[j] == "L" else nH
        d = nt / nm * seg
        d = min(d, per_layer_max)
        d = max(d, 0.0)
        thicknesses.append(d)

    # Enforce total <= max_total_nm.
    total = sum(thicknesses)
    if total > max_total_nm:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]

    # Enforce per-layer cap after scaling.
    thicknesses = [min(max(t, 0.0), max_layer_nm) for t in thicknesses]

    # Drop zero-thickness layers? Keep them (allowed, 0 in [0,max]).
    return materials, thicknesses


def _nelder_mead(objective, x0, args=(), max_iter=200, tol=1e-8, deadline=None):
    """Simple Nelder-Mead. Returns (best_x, best_f)."""
    x = np.array(x0, dtype=float)
    n = len(x)
    simplex = [x.copy()]
    for i in range(n):
        xi = x.copy()
        xi[i] = xi[i] + (0.05 if xi[i] == 0.0 else 0.05 * xi[i])
        simplex.append(xi)
    simplex = [np.array(s) for s in simplex]
    f = [objective(s, *args) for s in simplex]

    for it in range(max_iter):
        if deadline is not None and _elapsed() > deadline:
            break
        order = np.argsort(f)
        simplex = [simplex[i] for i in order]
        f = [f[i] for i in order]
        if abs(f[0] - f[-1]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])  # reflection
        fr = objective(xr, *args)
        if f[0] <= fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
        elif fr < f[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])  # expansion
            fe = objective(xe, *args)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)  # contraction
            fc = objective(xc, *args)
            if fc < f[-1]:
                simplex[-1] = xc
                f[-1] = fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f[i] = objective(simplex[i], *args)
    order = np.argsort(f)
    return simplex[order[0]], f[order[0]]


def design(problem):
    n_air = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]

    # 401 equally spaced wavelengths in [400,800]
    wavelengths = np.linspace(band[0], band[1], 401)

    # Objective for a discrete design: mean reflectance in percent.
    def obj(materials, thicknesses):
        return _score(materials, thicknesses, problem, wavelengths)

    # ----- Stage 1: coarse search over profile shapes, shape param, layer count.
    best_materials = None
    best_thicknesses = None
    best_score = float("inf")

    shapes = ["poly", "sqrt", "log", "smoothstep"]
    a_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    layer_counts = [3, 4, 5, 6, 8, 10, 12, 15, 18, 20]

    for nl in layer_counts:
        if _elapsed() > _DEADLINE - 30.0:
            break
        for shape in shapes:
            for a in a_vals:
                if _elapsed() > _DEADLINE - 30.0:
                    break
                materials, thicknesses = _make_design(shape, a, nl, problem)
                if len(materials) > max_layers:
                    continue
                if sum(thicknesses) > max_total_nm + 1e-6:
                    continue
                if any(t < -1e-9 or t > max_layer_nm + 1e-6 for t in thicknesses):
                    continue
                s = obj(materials, thicknesses)
                if s < best_score:
                    best_score = s
                    best_materials = list(materials)
                    best_thicknesses = list(thicknesses)

    # ----- Stage 2: material-pattern swap refinement (local search over patterns).
    # For the best design, try flipping individual layers between L and H and
    # re-optimizing thicknesses via Nelder-Mead.
    if best_materials is not None and _elapsed() < _DEADLINE - 20.0:
        cur_mats = list(best_materials)
        cur_th = list(best_thicknesses)

        def thickness_obj(th, mats):
            return _score(mats, list(th), problem, wavelengths)

        # Polish current design thicknesses.
        x0 = np.array(cur_th)
        best_x, best_f = _nelder_mead(
            thickness_obj, x0, args=(cur_mats,),
            max_iter=300, tol=1e-9, deadline=_DEADLINE - 15.0)
        best_f = min(best_f, obj(cur_mats, list(best_x)))
        if best_f < best_score:
            best_score = best_f
            best_thicknesses = list(np.clip(best_x, 0.0, max_layer_nm))
            # re-clamp total
            tot = sum(best_thicknesses)
            if tot > max_total_nm:
                sc = max_total_nm / tot
                best_thicknesses = [t * sc for t in best_thicknesses]

        # Pattern swaps: for each layer index, try flipping and polishing.
        n = len(cur_mats)
        for i in range(n):
            if _elapsed() > _DEADLINE - 15.0:
                break
            trial_mats = list(cur_mats)
            trial_mats[i] = "H" if trial_mats[i] == "L" else "L"
            # polish thicknesses for this pattern
            x0 = np.array(best_thicknesses)
            bx, bf = _nelder_mead(
                thickness_obj, x0, args=(trial_mats,),
                max_iter=150, tol=1e-9, deadline=_DEADLINE - 12.0)
            bf = min(bf, _score(trial_mats, list(bx), problem, wavelengths))
            if bf < best_score:
                best_score = bf
                cur_mats = trial_mats
                best_materials = list(trial_mats)
                best_thicknesses = list(np.clip(bx, 0.0, max_layer_nm))
                tot = sum(best_thicknesses)
                if tot > max_total_nm:
                    sc = max_total_nm / tot
                    best_thicknesses = [t * sc for t in best_thicknesses]

    # ----- Final safety: ensure constraints.
    if best_materials is None:
        # Fallback: single quarter-wave L layer centred in band.
        centre = 0.5 * (band[0] + band[1])
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * nL)]
    else:
        # clamp
        best_thicknesses = [min(max(t, 0.0), max_layer_nm) for t in best_thicknesses]
        tot = sum(best_thicknesses)
        if tot > max_total_nm:
            sc = max_total_nm / tot
            best_thicknesses = [t * sc for t in best_thicknesses]
        # trim to max_layers
        if len(best_materials) > max_layers:
            best_materials = best_materials[:max_layers]
            best_thicknesses = best_thicknesses[:max_layers]

    return best_materials, best_thicknesses
