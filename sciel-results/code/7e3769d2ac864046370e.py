# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using quadrature at M heights.
    x: array of shape (n,) with base positions.
    Returns: area (float).
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n
    
    L_sorted = np.sort(L, axis=0)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]
    union_lengths = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_fast(x, n=64, M=20000):
    """
    Optimized area computation using the identity:
        union_length = max(L) - min(L) + w*(n-1) - sum_j max(gap_j, w)
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n

    L_sorted = np.sort(L, axis=0)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]
    union_lengths = w + (L_sorted[-1, :] - L_sorted[0, :]) - np.sum(np.maximum(diffs - w[None, :], 0.0), axis=0)

    area = np.mean(union_lengths)
    return area


def _compute_area_softmin(x, n=64, M=20000, beta=50.0):
    """
    Smooth surrogate for the union area using soft-min/soft-max.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y) / n

    bL = beta * L
    bL_max = np.max(bL, axis=0, keepdims=True)
    logsumexp_max = np.log(np.sum(np.exp(bL - bL_max), axis=0, keepdims=True)) + bL_max
    smax_L = logsumexp_max[0, :] / beta

    bL_neg = -beta * L
    bL_neg_max = np.max(bL_neg, axis=0, keepdims=True)
    logsumexp_min = np.log(np.sum(np.exp(bL_neg - bL_neg_max), axis=0, keepdims=True)) + bL_neg_max
    smin_L = -logsumexp_min[0, :] / beta

    L_sorted = np.sort(L, axis=0)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]

    bd = beta * diffs
    bw = beta * w[None, :]
    tmp = np.stack([bd, bw], axis=0)
    tmp_max = np.max(tmp, axis=0, keepdims=True)
    logsumexp_smax = np.log(np.sum(np.exp(tmp - tmp_max), axis=0, keepdims=True)) + tmp_max
    smax_gaps = logsumexp_smax[0, :, :] / beta

    sum_smax_gaps = np.sum(smax_gaps, axis=0)

    union_lengths = w + (smax_L - smin_L) - sum_smax_gaps
    area = np.mean(union_lengths)
    return area


def _keich_initial(n=64):
    """
    Keich construction for initial positions.
    """
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _breakpoint_cd_1d(x, i, n=64, M=20000, deadline=None):
    """
    Exact 1-D coordinate descent for coordinate i using breakpoint analysis.
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M

    x_i_current = x[i]
    j_idx = np.arange(n)
    j_idx = j_idx[j_idx != i]
    a_j = a[j_idx]
    x_j = x[j_idx]

    diff_a = a_j - a[i]
    breakpoints = x_j[:, None] + diff_a[:, None] * y[None, :]

    lo_bound = x_i_current - 2.0
    hi_bound = x_i_current + 2.0
    mask = (breakpoints >= lo_bound) & (breakpoints <= hi_bound)
    bp_filtered = breakpoints[mask]

    if len(bp_filtered) == 0:
        test_pts = np.array([x_i_current - 0.1, x_i_current, x_i_current + 0.1])
        areas = []
        for tp in test_pts:
            x_trial = x.copy()
            x_trial[i] = tp
            areas.append(_compute_area_fast(x_trial, n, M))
        best_k = int(np.argmin(areas))
        new_xi = test_pts[best_k]
        return new_xi, areas[best_k]

    bp_sorted = np.unique(np.round(bp_filtered, decimals=10))

    candidates = list(bp_sorted)
    if len(bp_sorted) > 1:
        mids = (bp_sorted[:-1] + bp_sorted[1:]) / 2.0
        candidates.extend(mids.tolist())
    candidates.append(lo_bound)
    candidates.append(hi_bound)
    candidates.append(x_i_current)

    candidates = np.array(candidates)
    candidates = np.unique(np.round(candidates, decimals=10))

    best_val = np.inf
    best_xi = x_i_current
    for tp in candidates:
        if deadline is not None and time.time() > deadline:
            break
        x_trial = x.copy()
        x_trial[i] = tp
        val = _compute_area_fast(x_trial, n, M)
        if val < best_val:
            best_val = val
            best_xi = tp

    return best_xi, best_val


def _coordinate_descent(x, n=64, M=20000, max_iters=500, tol=1e-12, deadline=None):
    """
    Coordinate descent using exact breakpoint-aware 1-D minimization.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break

            new_xi, new_area = _breakpoint_cd_1d(x, i, n, M, deadline)

            if new_area < current_area - tol:
                x[i] = new_xi
                current_area = new_area
                improved = True

        if not improved:
            break

        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()

    return best_x, best_area


def _nelder_mead_softmin(x, n=64, M=20000, deadline=None, max_iter=200):
    """
    Nelder-Mead on the smooth soft-min surrogate.
    """
    from scipy.optimize import minimize

    beta = 50.0

    def objective(x_vec):
        return _compute_area_softmin(x_vec, n, M, beta=beta)

    x0 = x.copy()
    res = minimize(
        objective, x0, method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    x_nm = res.x
    area_exact = _compute_area_fast(x_nm, n, M)
    return x_nm, area_exact


def _simulated_annealing(x, n=64, M=20000, deadline=None, max_steps=50000):
    """
    Simulated annealing to escape local minima.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    T = 0.01
    cooling = 0.999

    rng = np.random.default_rng(42)

    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break

        n_perturb = int(rng.integers(1, 4))
        indices = rng.choice(n, size=n_perturb, replace=False)
        x_trial = x.copy()
        x_trial[indices] += rng.normal(0, T, size=n_perturb)

        trial_area = _compute_area_fast(x_trial, n, M)

        delta = trial_area - current_area
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_trial
            current_area = trial_area
            if current_area < best_area:
                best_area = current_area
                best_x = x.copy()

        T *= cooling
        step += 1

    return best_x, best_area


def _high_precision_polish(x, n=64, M=20000, deadline=None, max_iters=200):
    """
    High-precision polish: use tighter tolerance and more iterations.
    Also try smaller perturbations around the current point.
    """
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area
    
    # First, do standard CD with very tight tolerance
    x_cd, area_cd = _coordinate_descent(x, n, M, max_iters=max_iters, tol=1e-15, deadline=deadline)
    if area_cd < best_area:
        best_area = area_cd
        best_x = x_cd.copy()
    
    # Then, try local perturbations around the best point
    rng = np.random.default_rng(123)
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        
        # Small random perturbation
        x_trial = best_x + rng.normal(0, 1e-6, size=n)
        trial_area = _compute_area_fast(x_trial, n, M)
        
        if trial_area < best_area:
            best_area = trial_area
            best_x = x_trial.copy()
            # Polish from new point
            x_cd, area_cd = _coordinate_descent(best_x, n, M, max_iters=50, tol=1e-15, deadline=deadline)
            if area_cd < best_area:
                best_area = area_cd
                best_x = x_cd.copy()
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    """
    n = 64
    M = 20000

    start_time = time.time()
    deadline = start_time + 600.0

    x = _keich_initial(n)
    current_area = _compute_area_fast(x, n, M)
    best_x = x.copy()
    best_area = current_area

    # Phase 1: Breakpoint-aware coordinate descent
    x_cd, area_cd = _coordinate_descent(x, n, M, max_iters=100, tol=1e-12, deadline=deadline - 60)
    if area_cd < best_area:
        best_area = area_cd
        best_x = x_cd.copy()

    # Phase 2: Nelder-Mead on smooth surrogate
    if time.time() < deadline - 60:
        x_nm, area_nm = _nelder_mead_softmin(best_x, n, M, deadline=deadline - 30, max_iter=150)
        if area_nm < best_area:
            best_area = area_nm
            best_x = x_nm.copy()

    # Phase 3: Coordinate descent refinement
    if time.time() < deadline - 30:
        x_cd2, area_cd2 = _coordinate_descent(best_x, n, M, max_iters=100, tol=1e-12, deadline=deadline - 20)
        if area_cd2 < best_area:
            best_area = area_cd2
            best_x = x_cd2.copy()

    # Phase 4: Simulated annealing
    if time.time() < deadline - 20:
        x_sa, area_sa = _simulated_annealing(best_x, n, M, deadline=deadline - 10, max_steps=15000)
        if area_sa < best_area:
            best_area = area_sa
            best_x = x_sa.copy()

    # Phase 5: High-precision polish
    if time.time() < deadline - 5:
        x_final, area_final = _high_precision_polish(best_x, n, M, deadline=deadline - 1, max_iters=100)
        if area_final < best_area:
            best_area = area_final
            best_x = x_final.copy()

    if not np.all(np.isfinite(best_x)) or best_area <= 0:
        return _keich_initial(n)

    return best_x


# EVOLVE-BLOCK-END
