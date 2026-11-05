"""Broadband antireflection coating design via coordinate descent + Nelder-Mead."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _compute_mean_reflectance(labels, thicknesses, n0, ns, nL, nH, wl_grid):
    """Mean normal-incidence reflectance in percent over wl_grid (nm).
    Uses characteristic-matrix method. Returns float (percent)."""
    n_layers = len(labels)
    n_wl = len(wl_grid)
    # Precompute per-layer n
    n_arr = np.empty(n_layers)
    for i, lab in enumerate(labels):
        n_arr[i] = nL if lab == 'L' else nH
    d_arr = np.asarray(thicknesses, dtype=float)

    R_sum = 0.0
    for wl in wl_grid:
        # Build total matrix
        m11 = 1.0 + 0.0j
        m12 = 0.0 + 0.0j
        m21 = 0.0 + 0.0j
        m22 = 1.0 + 0.0j
        for i in range(n_layers):
            n = n_arr[i]
            d = d_arr[i]
            beta = 2.0 * math.pi * n * d / wl
            cb = math.cos(beta)
            sb = math.sin(beta)
            # M = [[cb, sb/(i n)], [i n sb, cb]]
            a = cb
            b = (sb / n) / 1j
            c = (1j) * n * sb
            dd = cb
            # multiply (m11,m12;m21,m22) by M
            n11 = m11 * a + m12 * c
            n12 = m11 * b + m12 * dd
            n21 = m21 * a + m22 * c
            n22 = m21 * b + m22 * dd
            m11, m12, m21, m22 = n11, n12, n21, n22
        # Admittance Y = B / A with B = 1
        Y = 1.0 / m11
        # Reflectance
        r = (n0 - Y) / (n0 + Y)
        R = (r * r.conjugate()).real
        R_sum += R
    return 100.0 * R_sum / n_wl


def _is_valid(labels, thicknesses, max_layers, max_total, max_layer):
    if len(labels) > max_layers:
        return False
    if len(labels) != len(thicknesses):
        return False
    total = 0.0
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
        total += max(0.0, t)
    if total > max_total + 1e-6:
        return False
    return True


def _nm_optimize(labels, thicknesses, n0, ns, nL, nH, wl_grid, max_layer, max_total,
                 max_iter=200, tol=1e-5, deadline=None):
    """Nelder-Mead on thicknesses (clamped to [0, max_layer], sum<=max_total)."""
    n = len(thicknesses)
    # initial simplex
    simplex = []
    f0 = _compute_mean_reflectance(labels, thicknesses, n0, ns, nL, nH, wl_grid)
    simplex.append((list(thicknesses), f0))
    for i in range(n):
        pt = list(thicknesses)
        step = 10.0 if thicknesses[i] < 10.0 else 0.1 * thicknesses[i]
        pt[i] = min(max_layer, max(0.0, pt[i] + step))
        f = _compute_mean_reflectance(labels, pt, n0, ns, nL, nH, wl_grid)
        simplex.append((pt, f))

    best_pt = simplex[0][0]
    best_f = f0

    it = 0
    while it < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        # sort
        simplex.sort(key=lambda x: x[1])
        # update best
        if simplex[0][1] < best_f:
            best_f = simplex[0][1]
            best_pt = list(simplex[0][0])
        # check convergence
        fs = [s[1] for s in simplex]
        if (max(fs) - min(fs)) < tol:
            break
        # worst point
        worst = simplex[-1]
        worst_pt, worst_f = worst
        # centroid of all but worst
        cent = [0.0] * n
        for s in simplex[:-1]:
            for i in range(n):
                cent[i] += s[0][i]
        for i in range(n):
            cent[i] /= (n)
        # reflection
        alpha = 1.0
        refl = [cent[i] + alpha * (cent[i] - worst_pt[i]) for i in range(n)]
        refl = [min(max_layer, max(0.0, v)) for v in refl]
        # enforce total
        tot = sum(refl)
        if tot > max_total:
            scale = max_total / tot
            refl = [v * scale for v in refl]
        f_refl = _compute_mean_reflectance(labels, refl, n0, ns, nL, nH, wl_grid)
        if f_refl < simplex[0][1]:
            # expansion
            gamma = 2.0
            exp = [cent[i] + gamma * (cent[i] - worst_pt[i]) for i in range(n)]
            exp = [min(max_layer, max(0.0, v)) for v in exp]
            tot = sum(exp)
            if tot > max_total:
                scale = max_total / tot
                exp = [v * scale for v in exp]
            f_exp = _compute_mean_reflectance(labels, exp, n0, ns, nL, nH, wl_grid)
            if f_exp < f_refl:
                simplex[-1] = (exp, f_exp)
            else:
                simplex[-1] = (refl, f_refl)
        elif f_refl < simplex[-2][1]:
            simplex[-1] = (refl, f_refl)
        else:
            # contraction
            rho = 0.5
            cont = [cent[i] + rho * (worst_pt[i] - cent[i]) for i in range(n)]
            cont = [min(max_layer, max(0.0, v)) for v in cont]
            tot = sum(cont)
            if tot > max_total:
                scale = max_total / tot
                cont = [v * scale for v in cont]
            f_cont = _compute_mean_reflectance(labels, cont, n0, ns, nL, nH, wl_grid)
            if f_cont < worst_f:
                simplex[-1] = (cont, f_cont)
            else:
                # shrink
                for k in range(1, len(simplex)):
                    for i in range(n):
                        simplex[k][0][i] = simplex[0][0][i] + 0.5 * (simplex[k][0][i] - simplex[0][0][i])
                    simplex[k] = (simplex[k][0], _compute_mean_reflectance(labels, simplex[k][0], n0, ns, nL, nH, wl_grid))
        it += 1
    return best_pt, best_f


def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band0, band1 = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    # wavelength grid: 401 equally spaced points
    wl_grid = np.linspace(band0, band1, 401)

    deadline = time.time() + 110.0  # safety margin before 120s

    # Try several layer counts; alternate L/H starting with L (air side)
    best_labels = None
    best_thicknesses = None
    best_f = float('inf')

    # candidate layer counts (alternating, starting L)
    for n_layers in [4, 6, 8, 10, 12, 14]:
        if n_layers > max_layers:
            continue
        labels = []
        for i in range(n_layers):
            labels.append('L' if i % 2 == 0 else 'H')
        # initial guess: quarter-wave at band center
        centre = 0.5 * (band0 + band1)
        init = []
        for lab in labels:
            n = nL if lab == 'L' else nH
            init.append(centre / (4.0 * n))
        # clamp to max_layer
        init = [min(max_layer, max(0.0, t)) for t in init]
        # scale to fit total
        tot = sum(init)
        if tot > max_total:
            scale = max_total / tot
            init = [t * scale for t in init]

        if time.time() > deadline - 15.0:
            break

        # coordinate descent first (fast, few evals)
        pt = list(init)
        for cd_iter in range(30):
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(len(pt)):
                # try small perturbations
                base = pt[i]
                for delta in [0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
                    cand = list(pt)
                    cand[i] = min(max_layer, max(0.0, base + delta))
                    tot = sum(cand)
                    if tot > max_total:
                        scale = max_total / tot
                        cand = [v * scale for v in cand]
                    f = _compute_mean_reflectance(labels, cand, n0, ns, nL, nH, wl_grid)
                    if f < best_f:
                        best_f = f
                        best_labels = list(labels)
                        best_thicknesses = list(cand)
                    if f < _compute_mean_reflectance(labels, pt, n0, ns, nL, nH, wl_grid):
                        pt = cand
                        improved = True
                        break
            if not improved:
                break

        # Nelder-Mead polish
        if time.time() > deadline - 10.0:
            break
        pt2, f2 = _nm_optimize(labels, pt, n0, ns, nL, nH, wl_grid,
                               max_layer, max_total, max_iter=150,
                               tol=1e-4, deadline=deadline - 5.0)
        if f2 < best_f:
            best_f = f2
            best_labels = list(labels)
            best_thicknesses = list(pt2)

    # fallback: single L quarter-wave
    if best_labels is None:
        centre = 0.5 * (band0 + band1)
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * nL)]

    # final clamp
    for i in range(len(best_thicknesses)):
        best_thicknesses[i] = min(max_layer, max(0.0, best_thicknesses[i]))
    tot = sum(best_thicknesses)
    if tot > max_total:
        scale = max_total / tot
        best_thicknesses = [t * scale for t in best_thicknesses]

    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
