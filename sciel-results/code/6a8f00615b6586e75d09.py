# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N, dtype=np.float64)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

# Precompute bin indices for each projection
_IDX_D = (_XS - _YS).astype(np.int64) + 9
_IDX_H0 = _XS.astype(np.int64)
_IDX_H1 = (_XS + _YS).astype(np.int64)
_IDX_H2 = (_XS + 2 * _YS).astype(np.int64)

_N_BINS_D = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28


def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector, ignoring zeros."""
    s = counts.sum()
    if s <= 1e-30:
        return 0.0
    p = counts / s
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_value(P):
    """Compute D/M ratio for a given 10x10 distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0
    p = p / total

    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m


def _soft_max(values, temperature=0.1):
    """Smooth approximation of max using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    log_sum_exp = max_val + np.log(exp_scaled.sum())
    return float(log_sum_exp * temperature)


def _negative_value_smooth(theta, temperature=0.1):
    """Objective to minimize: -D/M_smooth. Uses soft-max for M."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()

    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m_smooth = _soft_max(np.array([h0, h1, h2]), temperature)
    if m_smooth <= 1e-9:
        return 1e6
    return -(hd / m_smooth)


def _negative_value(theta):
    """Objective to minimize: -D/M. theta is a 100-dim vector of log-probabilities."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    if val <= 1e-9:
        return 1e6
    return -val


def _uniform_init():
    return np.zeros(_N * _N, dtype=np.float64)


def _random_init():
    return np.random.randn(_N * _N) * 0.1


def _theta_to_P(theta):
    """Convert log-probabilities to a normalized 10x10 probability matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s > 1e-9:
        P = P / s
    return P


def _structured_init_diagonal():
    """Initialize with mass concentrated on diagonals (x - y = constant)."""
    theta = np.zeros(_N * _N)
    for d in range(-4, 5):
        for y in range(_N):
            x = y + d
            if 0 <= x < _N:
                theta[y * _N + x] = 1.0
    theta += np.random.randn(_N * _N) * 0.3
    return theta


def _structured_init_row():
    """Initialize with mass concentrated on a few rows."""
    theta = np.zeros(_N * _N)
    for y in range(_N):
        for x in range(_N):
            theta[y * _N + x] = 0.5
    theta += np.random.randn(_N * _N) * 0.5
    return theta


def _structured_init_column():
    """Initialize with mass concentrated on a few columns."""
    theta = np.zeros(_N * _N)
    for x in range(_N):
        for y in range(_N):
            theta[y * _N + x] = 0.5
    theta += np.random.randn(_N * _N) * 0.5
    return theta


def _structured_init_anti_diagonal():
    """Initialize with mass concentrated on anti-diagonals (x + y = constant)."""
    theta = np.zeros(_N * _N)
    for s in range(0, 18, 2):
        for y in range(_N):
            x = s - y
            if 0 <= x < _N:
                theta[y * _N + x] = 1.0
    theta += np.random.randn(_N * _N) * 0.3
    return theta


def _structured_init_sparse():
    """Initialize with mass on a sparse set of cells."""
    theta = np.full(_N * _N, -5.0)
    # Pick ~15 random cells
    indices = np.random.choice(_N * _N, size=15, replace=False)
    theta[indices] = 0.0
    theta += np.random.randn(_N * _N) * 0.2
    return theta


def _discrete_refine(P, max_iters=500, tol=1e-10):
    """
    Greedy discrete support refinement: iteratively zero out the cell with
    the smallest mass and re-optimize the remaining support via continuous
    optimization on the active set.
    """
    P = P.copy()
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return P
    P = P / s

    best_P = P.copy()
    best_score = _compute_value(P)

    active = np.ones(P.shape, dtype=bool)

    for iteration in range(max_iters):
        masked = np.where(active, P, np.inf)
        min_idx = np.unravel_index(np.argmin(masked), P.shape)
        min_val = P[min_idx]

        if min_val < 1e-12:
            break

        P_new = P.copy()
        P_new[min_idx] = 0.0
        active_new = active.copy()
        active_new[min_idx] = False

        if active_new.sum() < 3:
            break

        s_new = P_new.sum()
        if s_new <= 1e-9:
            break
        P_new = P_new / s_new

        try:
            active_indices = np.where(active_new)
            n_active = len(active_indices[0])

            if n_active < 2:
                break

            p_active = P_new[active_indices]
            p_active_safe = np.maximum(p_active, 1e-18)
            p_active_safe = p_active_safe / p_active_safe.sum()
            theta_active = np.log(p_active_safe)

            def obj_active(theta_a):
                theta_a = np.asarray(theta_a)
                theta_shifted = theta_a - theta_a.max()
                exp_theta = np.exp(theta_shifted)
                p_a = exp_theta / exp_theta.sum()

                P_full = np.zeros(P.shape)
                P_full[active_indices] = p_a
                return -_compute_value(P_full)

            result = minimize(
                obj_active,
                theta_active,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12}
            )

            theta_a_opt = result.x
            theta_shifted = theta_a_opt - theta_a_opt.max()
            exp_theta = np.exp(theta_shifted)
            p_a_opt = exp_theta / exp_theta.sum()

            P_opt = np.zeros(P.shape)
            P_opt[active_indices] = p_a_opt

            score_opt = _compute_value(P_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt.copy()
                P = P_opt.copy()
                active = active_new.copy()
        except Exception:
            score_new = _compute_value(P_new)
            if score_new > best_score:
                best_score = score_new
                best_P = P_new.copy()
                P = P_new.copy()
                active = active_new.copy()

    return best_P


def _pairwise_swap_optimize(P, max_swaps=200):
    """
    Try pairwise swaps: move mass from one cell to another and see if it improves.
    """
    P = P.copy()
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        return P
    P = P / s

    best_P = P.copy()
    best_score = _compute_value(P)

    for _ in range(max_swaps):
        improved = False
        # Try moving a small fraction of mass from each cell to each other cell
        for i in range(_N):
            for j in range(_N):
                if P[i, j] < 1e-10:
                    continue
                for i2 in range(_N):
                    for j2 in range(_N):
                        if i2 == i and j2 == j:
                            continue
                        # Try moving 10% of mass from (i,j) to (i2,j2)
                        P_new = P.copy()
                        amount = P[i, j] * 0.1
                        P_new[i, j] -= amount
                        P_new[i2, j2] += amount
                        P_new = np.clip(P_new, 0.0, None)
                        s_new = P_new.sum()
                        if s_new <= 1e-9:
                            continue
                        P_new = P_new / s_new
                        score = _compute_value(P_new)
                        if score > best_score + 1e-12:
                            best_score = score
                            best_P = P_new.copy()
                            P = P_new.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_P


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 585.0

    best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)
    best_score = _compute_value(best_P)

    np.random.seed(42)

    # Phase 1: Multiple restarts with smooth objective
    n_restarts_smooth = 30
    for i in range(n_restarts_smooth):
        if time.time() > deadline - 200.0:
            break

        if i == 0:
            theta0 = _uniform_init()
        elif i % 6 == 1:
            theta0 = _random_init()
        elif i % 6 == 2:
            theta0 = _structured_init_diagonal()
        elif i % 6 == 3:
            theta0 = _structured_init_row()
        elif i % 6 == 4:
            theta0 = _structured_init_column()
        elif i % 6 == 5:
            theta0 = _structured_init_anti_diagonal()
        else:
            theta0 = _structured_init_sparse()

        try:
            temp = 0.5 if i < 6 else (0.2 if i < 15 else 0.1)
            result = minimize(
                _negative_value_smooth,
                theta0,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'gtol': 1e-7, 'ftol': 1e-10}
            )
            P = _theta_to_P(result.x)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            continue

    # Phase 2: Discrete support refinement
    if time.time() < deadline - 150.0:
        try:
            P_refined = _discrete_refine(best_P, max_iters=300)
            score = _compute_value(P_refined)
            if score > best_score:
                best_score = score
                best_P = P_refined.copy()
        except Exception:
            pass

    # Phase 3: Pairwise swap optimization
    if time.time() < deadline - 100.0:
        try:
            P_swapped = _pairwise_swap_optimize(best_P, max_swaps=100)
            score = _compute_value(P_swapped)
            if score > best_score:
                best_score = score
                best_P = P_swapped.copy()
        except Exception:
            pass

    # Phase 4: Refine with exact objective
    if time.time() < deadline - 80.0:
        try:
            p_flat = best_P.ravel()
            p_safe = np.maximum(p_flat, 1e-15)
            theta_best = np.log(p_safe / p_safe.sum())

            result = minimize(
                _negative_value,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            P = _theta_to_P(result.x)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass

    # Phase 5: Another round of discrete refinement
    if time.time() < deadline - 50.0:
        try:
            P_refined = _discrete_refine(best_P, max_iters=200)
            score = _compute_value(P_refined)
            if score > best_score:
                best_score = score
                best_P = P_refined.copy()
        except Exception:
            pass

    # Phase 6: Another pairwise swap round
    if time.time() < deadline - 30.0:
        try:
            P_swapped = _pairwise_swap_optimize(best_P, max_swaps=50)
            score = _compute_value(P_swapped)
            if score > best_score:
                best_score = score
                best_P = P_swapped.copy()
        except Exception:
            pass

    # Phase 7: Final high-precision polish
    if time.time() < deadline - 15.0:
        try:
            p_flat = best_P.ravel()
            p_safe = np.maximum(p_flat, 1e-18)
            theta_best = np.log(p_safe / p_safe.sum())

            result = minimize(
                _negative_value,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            P = _theta_to_P(result.x)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass

    # Phase 8: Final discrete refinement
    if time.time() < deadline - 5.0:
        try:
            P_refined = _discrete_refine(best_P, max_iters=100)
            score = _compute_value(P_refined)
            if score > best_score:
                best_score = score
                best_P = P_refined.copy()
        except Exception:
            pass

    # Final cleanup
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    return best_P


# EVOLVE-BLOCK-END
