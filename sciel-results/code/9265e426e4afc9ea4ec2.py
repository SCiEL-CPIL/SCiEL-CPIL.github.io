"""Broadband antireflection coating design: exact characteristic-matrix reflectance
plus a two-stage (discrete layer-count search + thickness refinement) optimizer."""
import time
import numpy as np

_START = time.time()
_DEADLINE = 150.0 - 10.0  # safety margin before the 150 s wall-clock kill


def _reflectance_stack(materials, thicknesses, wl, na, ns):
    """Mean normal-incidence reflectance (percent) for a stack, exact 2x2 char matrix."""
    n = np.array([na] + list(materials) + [ns], dtype=float)
    d = np.array([0.0] + list(thicknesses) + [0.0], dtype=float)
    k = 2.0 * np.pi / wl
    m = np.empty((len(n) - 1, 2, 2), dtype=complex)
    for j in range(len(n) - 1):
        phase = 1j * k * n[j] * d[j]
        c = np.cos(phase)
        s = np.sin(phase)
        m[j, 0, 0] = c
        m[j, 0, 1] = 1j * s / n[j]
        m[j, 1, 0] = 1j * n[j] * s
        m[j, 1, 1] = c
    M = m[0].copy()
    for j in range(1, len(m)):
        M = M @ m[j]
    Y = M[1, 0] / M[0, 0]
    r = (na - Y) / (na + Y)
    return float(np.mean(np.abs(r) ** 2) * 100.0)


def _coarse_objective(materials, thicknesses, wl_coarse, na, ns):
    """Coarse-wavelength objective with a total-thickness penalty."""
    total = float(np.sum(thicknesses))
    pen = 0.0 if total <= 1000.0 else 1.0 + 0.01 * (total - 1000.0)
    return _reflectance_stack(materials, thicknesses, wl_coarse, na, ns) + pen


def _nelder_mead(f, x0, bounds, max_iter, tol=1e-7, max_time=None):
    """Bounded Nelder–Mead; returns the best (x, f) found."""
    n = len(x0)
    lo = np.array(bounds[0], dtype=float)
    hi = np.array(bounds[1], dtype=float)
    x = np.clip(np.array(x0, dtype=float), lo, hi)
    simplex = [x.copy()]
    fvals = [float(f(x))]
    for i in range(n):
        xk = x.copy()
        xk[i] = xk[i] + 0.25 * (hi[i] - lo[i])
        xk = np.clip(xk, lo, hi)
        simplex.append(xk)
        fvals.append(float(f(xk)))
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    it = 0
    while it < max_iter:
        if max_time is not None and time.time() > max_time:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = float(f(xr))
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = float(f(xe))
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = float(f(xc))
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = float(f(simplex[i]))
        it += 1
    order = int(np.argmin(fvals))
    return simplex[order], fvals[order]


def _simulate_annealing(counts, wl_coarse, na, ns, max_time):
    """Global search over layer counts (nL,nH); returns list of (nL,nH) candidates."""
    NMAX = 20
    nL = np.random.randint(0, NMAX + 1)
    nH = np.random.randint(0, NMAX + 1)
    while nL + nH > NMAX:
        nL = np.random.randint(0, NMAX + 1)
        nH = np.random.randint(0, NMAX + 1)
    best = (nL, nH)
    best_val = float("inf")
    cur_val = best_val
    T = 0.25
    candidates = []
    t0 = time.time()
    while time.time() - t0 < max_time:
        def val(nl, nh):
            if nl + nh == 0:
                return 4.26
            if nl + nh > NMAX:
                return 1e6
            if nl * 45.0 + nh * 26.0 > 1000.0:
                return 1e6
            mats = ["L"] * nl + ["H"] * nh
            th = [105.0] * nl + [62.0] * nh
            return _reflectance_stack(mats, th, wl_coarse, na, ns)

        cur_val = val(nL, nH)
        if cur_val < best_val - 1e-9:
            best_val = cur_val
            best = (nL, nH)
            candidates.append((nL, nH))
        # propose a move
        r = np.random.rand()
        if r < 0.45:
            nL = max(0, min(NMAX - nH, nL + np.random.choice([-2, -1, 1, 2])))
        elif r < 0.9:
            nH = max(0, min(NMAX - nL, nH + np.random.choice([-2, -1, 1, 2])))
        else:
            nL = np.random.randint(0, NMAX + 1)
            nH = np.random.randint(0, NMAX + 1)
            while nL + nH > NMAX:
                nL = np.random.randint(0, NMAX + 1)
                nH = np.random.randint(0, NMAX + 1)
        new_val = val(nL, nH)
        if new_val < cur_val or np.random.rand() < np.exp(-(new_val - cur_val) / max(T, 1e-6)):
            cur_val = new_val
        T = max(T * 0.99, 0.02)
    candidates.append(best)
    return candidates


def design(problem):
    global _START
    np.random.seed(12345)
    na = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    band = problem["band_nm"]
    wl = np.linspace(band[0], band[1], 401)
    wl_coarse = np.linspace(band[0], band[1], 21)

    # Baseline: single low-index quarter-wave layer (valid, ~4.26 % or better).
    lam_c = 0.5 * (band[0] + band[1])
    base_th = lam_c / (4.0 * nL)
    base_th = min(max(base_th, 0.0), 400.0)
    best_stack = (["L"], [base_th])
    best_score = _reflectance_stack(["L"], [base_th], wl, na, ns)

    def consider(mats, ths):
        nonlocal best_stack, best_score
        if len(ths) > 20:
            return
        if any(t < 0.0 or t > 400.0 for t in ths):
            return
        if sum(ths) > 1000.0 + 1e-6:
            return
        s = _reflectance_stack(mats, list(ths), wl, na, ns)
        if s < best_score - 1e-9:
            best_score = s
            best_stack = (list(mats), list(ths))

    consider(best_stack[0], best_stack[1])

    # ---- Stage A: global layer-count search on the coarse objective ----
    tA = time.time()
    sa_budget = max(2.0, min(12.0, _DEADLINE - tA))
    candidates = _simulate_annealing([nL, nH], wl_coarse, na, ns, sa_budget)

    # ---- Stage B: refine thicknesses for promising count structures ----
    tB = time.time()
    b_budget = max(2.0, _DEADLINE - tB)
    per = b_budget / max(1, len(candidates))

    for (nl, nh) in candidates:
        if time.time() > _DEADLINE - 3.0:
            break
        if nl + nh == 0:
            continue
        if nl * 45.0 + nh * 26.0 > 1000.0:
            continue
        mats = ["L"] * nl + ["H"] * nh
        x0 = np.array([105.0] * nl + [62.0] * nh, dtype=float)
        lo = np.zeros(len(x0))
        hi = np.full(len(x0), 400.0)
        f = lambda x, m=mats: _coarse_objective(m, x, wl_coarse, na, ns)
        x_best, _ = _nelder_mead(f, x0, (lo, hi), max_iter=400, max_time=time.time() + per)
        x_best = np.clip(x_best, 0.0, 400.0)
        consider(mats, list(x_best))
        # final exact-401-pt polish (only if still feasible and time remains)
        if time.time() < _DEADLINE - 3.0 and sum(x_best) <= 1000.0:
            f2 = lambda x, m=mats: _reflectance_stack(m, x, wl, na, ns)
            x_pol, _ = _nelder_mead(f2, x_best, (lo, hi), max_iter=120,
                                     max_time=min(time.time() + per * 0.5, _DEADLINE - 3.0))
            x_pol = np.clip(x_pol, 0.0, 400.0)
            consider(mats, list(x_pol))

    # Guarantee a valid return.
    if sum(best_stack[1]) > 1000.0:
        best_stack = (["L"], [base_th])
    return best_stack
