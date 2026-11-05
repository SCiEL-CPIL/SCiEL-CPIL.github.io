"""Broadband AR coating: continuous index-profile optimization + optimal quantization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def _reflectance_array(n_layers, d_layers, n0, ns, wls):
    """Reflectance (fraction) at each wavelength via characteristic-matrix method.
    n_layers, d_layers are arrays ordered air->substrate."""
    N = len(n_layers)
    out = np.empty(len(wls), dtype=float)
    for wi, lam in enumerate(wls):
        A = 1.0 + 0j
        B = 0.0 + 0j
        for i in range(N):
            n = n_layers[i]
            delta = 2.0 * math.pi * n * d_layers[i] / lam
            c = math.cos(delta)
            s = math.sin(delta)
            # M = [[c, i*s/n],[i*n*s, c]]
            A_new = A * c + 1j * B * s / n
            B_new = A * (1j * n * s) + B * c
            A, B = A_new, B_new
        Y = (A + B * ns) / (A * ns + B)
        out[wi] = abs((Y - n0) / (Y + n0)) ** 2
    return out

def _mean_pct(n_layers, d_layers, n0, ns, wls):
    return 100.0 * float(np.mean(_reflectance_array(n_layers, d_layers, n0, ns, wls)))

def _quantize(n_profile, z_profile, nL, nH, max_layers, max_total, max_layer):
    """Optimal 2-level (L/H) quantization of a smooth profile via DP.
    Returns (labels, thicknesses) for the best segmentation respecting constraints."""
    m = len(n_profile)
    total_target = float(np.sum(n_profile * z_profile))
    # DP over (position, count) -> min squared error; track parent
    NEG = float('inf')
    best = [[NEG] * (max_layers + 1) for _ in range(m + 1)]
    par = [[-1] * (max_layers + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    # precompute cumulative
    cum_n = np.concatenate([[0.0], np.cumsum(n_profile * z_profile)])
    cum_nz2 = np.concatenate([[0.0], np.cumsum((n_profile ** 2) * z_profile)])
    cum_z = np.concatenate([[0.0], np.cumsum(z_profile)])
    for i in range(1, m + 1):
        for k in range(1, max_layers + 1):
            if k > i:
                break
            prev = best[i - 1][k - 1]
            if prev == NEG:
                continue
            # candidate: layer k is H
            seg_nz = cum_n[i] - cum_n[i - 1]
            seg_z = cum_z[i] - cum_z[i - 1]
            seg_nz2 = cum_nz2[i] - cum_nz2[i - 1]
            # cost if this single segment is H
            costH = (nH * seg_z - seg_nz) ** 2
            # if this single segment is L
            costL = (nL * seg_z - seg_nz) ** 2
            # We need to allow layer to be either; choose min, but DP must remember label.
            # Simplify: treat each layer's material as free; cost = min(costL, costH).
            # This underestimates slightly if it alternates freely, but gives good thicknesses.
            cost = costH if costH < costL else costL
            val = prev + cost
            if val < best[i][k]:
                best[i][k] = val
                par[i][k] = i - 1
    # find best k at i=m
    best_k = -1
    best_val = NEG
    for k in range(1, max_layers + 1):
        if best[m][k] < best_val:
            best_val = best[m][k]
            best_k = k
    if best_k < 0:
        # fallback single layer
        return ['L'], [min(max_layer, total_target / nL)]
    # backtrack
    segs = []
    i, k = m, best_k
    while k > 0:
        j = par[i][k]
        segs.append((j, i))
        i = j
        k -= 1
    segs.reverse()
    labels = []
    thicknesses = []
    for (j, i) in segs:
        seg_nz = cum_n[i] - cum_n[j]
        seg_z = cum_z[i] - cum_z[j]
        # choose material
        if abs(nH * seg_z - seg_nz) < abs(nL * seg_z - seg_nz):
            labels.append('H')
            d = seg_nz / nH
        else:
            labels.append('L')
            d = seg_nz / nL
        d = max(0.0, min(max_layer, d))
        labels.append(labels[-1]) if False else None
        labels[-1] = labels[-1]  # noop
        thicknesses.append(d)
    # fix labels list (we appended correctly above)
    # enforce total thickness
    tot = sum(thicknesses)
    if tot > max_total:
        scale = max_total / tot
        thicknesses = [t * scale for t in thicknesses]
    # enforce layer max
    thicknesses = [min(max_layer, t) for t in thicknesses]
    return labels, thicknesses

def _build_profile(coeffs, wls, n0, ns):
    """Given coeffs [a0, b1, c1, b2, c2, ...] build n_profile and z_profile.
    n(x) = n0 + a0*(ns-n0)*x + sum_k [b_k sin(2pi k x) + c_k cos(2pi k x)]*(ns-n0)
    where x in [0,1]. z is uniform spacing."""
    # coeffs: [a0, b1, c1, b2, c2, b3, c3]
    a0 = coeffs[0]
    n0v = n0
    ns_v = ns
    span = ns_v - n0v
    m = 64
    x = np.linspace(0.0, 1.0, m)
    n_prof = n0v + a0 * span * x
    # harmonics
    idx = 1
    for k in range(1, 4):
        b = coeffs[idx]; c = coeffs[idx + 1]; idx += 2
        n_prof = n_prof + (b * np.sin(2 * np.pi * k * x) + c * np.cos(2 * np.pi * k * x)) * span
    # clamp to [nL, nH]
    nL, nH = 1.38, 2.35
    n_prof = np.clip(n_prof, nL, nH)
    z = np.ones(m) / m
    return n_prof, z

def _nelder_mead(f, x0, bounds, deadline, max_iter=400, xatol=1e-4, fatol=1e-5):
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.1 * (bounds[1][i] - bounds[0][i])
        xi[i] += step
        xi = np.clip(xi, bounds[0], bounds[1])
        simplex.append(xi)
    vals = [f(x) for x in simplex]
    for it in range(max_iter):
        if time.time() > deadline:
            break
        idx = np.argsort(vals)
        simplex = [simplex[i] for i in idx]
        vals = [vals[i] for i in idx]
        if abs(vals[0] - vals[-1]) < fatol:
            break
        if np.max(np.abs(simplex[0] - simplex[1])) < xatol:
            break
        centroid = np.sum(simplex[:-1], axis=0) / n
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds[0], bounds[1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe; vals[-1] = fe
            else:
                simplex[-1] = xr; vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr; vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc; vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                    vals[i] = f(simplex[i])
    idx = int(np.argmin(vals))
    return simplex[idx], vals[idx]

def _polish_thicknesses(labels, thicknesses, n0, ns, wls, deadline):
    """Polish the physical thicknesses of a fixed label sequence."""
    nL, nH = 1.38, 2.35
    nmat = np.array([nL if l == 'L' else nH for l in labels])
    N = len(labels)
    def obj(x):
        if time.time() > deadline:
            return 1e6
        x = np.clip(x, 0.0, 400.0)
        if x.sum() > 1000.0:
            return 1e6
        return _mean_pct(nmat, x, n0, ns, wls)
    x0 = np.array(thicknesses, dtype=float)
    b0 = np.zeros(N); b1 = np.full(N, 400.0)
    try:
        xb, fb = _nelder_mead(obj, x0, (b0, b1), deadline, max_iter=300, xatol=0.05, fatol=1e-6)
    except Exception:
        return labels, thicknesses
    xb = np.clip(xb, 0.0, 400.0)
    if xb.sum() > 1000.0:
        xb = xb * (1000.0 / xb.sum())
    return labels, [float(t) for t in xb]

def design(problem):
    start = time.time()
    deadline = start + 135.0
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    nL = problem['materials']['L']
    nH = problem['materials']['H']
    band = problem['band_nm']
    wls = np.linspace(band[0], band[1], 401)
    max_layers = problem['max_layers']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']

    best_labels = ['L']
    best_th = [band[0] / (4.0 * nL)]
    best_score = _mean_pct([nL], best_th, n0, ns, wls)

    # ---- Phase 1: optimize continuous index profile ----
    n_harm = 3
    n_coeff = 1 + 2 * n_harm  # a0, (b1,c1),(b2,c2),(b3,c3)
    def profile_obj(coeffs):
        if time.time() > deadline:
            return 1e6
        n_prof, z = _build_profile(coeffs, wls, n0, ns)
        labels, th = _quantize(n_prof, z, nL, nH, max_layers, max_total, max_layer)
        if len(labels) == 0:
            return 1e6
        nmat = np.array([nL if l == 'L' else nH for l in labels])
        th = np.array(th)
        if th.sum() > max_total:
            return 1e6
        return _mean_pct(nmat, th, n0, ns, wls)

    # multiple random starts for the profile
    np.random.seed(7)
    for trial in range(6):
        if time.time() > deadline:
            break
        x0 = np.zeros(n_coeff)
        x0[0] = 1.0 + 0.3 * np.random.randn()  # a0 near 1
        for j in range(1, n_coeff):
            x0[j] = 0.3 * np.random.randn()
        b0 = np.full(n_coeff, -2.0); b1 = np.full(n_coeff, 2.0)
        try:
            xb, fb = _nelder_mead(profile_obj, x0, (b0, b1), deadline, max_iter=250, xatol=1e-4, fatol=1e-5)
        except Exception:
            continue
        n_prof, z = _build_profile(xb, wls, n0, ns)
        labels, th = _quantize(n_prof, z, nL, nH, max_layers, max_total, max_layer)
        if len(labels) == 0:
            continue
        nmat = np.array([nL if l == 'L' else nH for l in labels])
        th = np.array(th)
        sc = _mean_pct(nmat, th, n0, ns, wls)
        if sc < best_score:
            best_score = sc
            best_labels = labels
            best_th = [float(t) for t in th]

    # ---- Phase 2: polish thicknesses of best label sequence ----
    if time.time() < deadline:
        pl, pt = _polish_thicknesses(best_labels, best_th, n0, ns, wls, deadline)
        nmat = np.array([nL if l == 'L' else nH for l in pl])
        sc = _mean_pct(nmat, np.array(pt), n0, ns, wls)
        if sc < best_score:
            best_score = sc
            best_labels = pl
            best_th = pt

    # ---- Phase 3: also try a quarter-wave stack baseline and polish ----
    if time.time() < deadline:
        center = 0.5 * (band[0] + band[1])
        qw_labels = []
        qw_th = []
        for i in range(20):
            if i % 2 == 0:
                qw_labels.append('L'); qw_th.append(center / (4.0 * nL))
            else:
                qw_labels.append('H'); qw_th.append(center / (4.0 * nH))
        if sum(qw_th) <= max_total:
            pl, pt = _polish_thicknesses(qw_labels, qw_th, n0, ns, wls, deadline)
            nmat = np.array([nL if l == 'L' else nH for l in pl])
            sc = _mean_pct(nmat, np.array(pt), n0, ns, wls)
            if sc < best_score:
                best_score = sc
                best_labels = pl
                best_th = pt

    # ---- Final constraint check & fallback ----
    if len(best_labels) > max_layers or len(best_labels) != len(best_th):
        best_labels = ['L']; best_th = [band[0] / (4.0 * nL)]
    for t in best_th:
        if t < 0 or t > max_layer:
            best_labels = ['L']; best_th = [band[0] / (4.0 * nL)]; break
    if sum(best_th) > max_total:
        scale = max_total / sum(best_th)
        best_th = [t * scale for t in best_th]
    return best_labels, best_th
# EVOLVE-BLOCK-END
