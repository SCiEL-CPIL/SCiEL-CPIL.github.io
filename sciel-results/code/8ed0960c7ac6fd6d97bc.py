# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9, but for pooling we can use as-is with appropriate binning


def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())


def _compute_value(P):
    """Compute the ratio D/M for a given distribution P."""
    p = P.ravel()
    
    # D: distribution of x - y (slope -1)
    # x - y ranges from -9 to 9, so 19 possible values
    d_bins = np.zeros(19)
    np.add.at(d_bins, _XM1Y + 9, p)
    hd = _entropy(d_bins)
    
    # M: max of H_0, H_1, H_2
    # H_0: distribution of x (10 values)
    h0_bins = np.zeros(_N)
    np.add.at(h0_bins, _XS, p)
    hm0 = _entropy(h0_bins)
    
    # H_1: distribution of x + y (ranges 0 to 18, so 19 values)
    h1_bins = np.zeros(2 * _N - 1)
    np.add.at(h1_bins, _XY, p)
    hm1 = _entropy(h1_bins)
    
    # H_2: distribution of x + 2y (ranges 0 to 27, so 28 values)
    h2_bins = np.zeros(2 * _N + _N - 1)  # 0 to 27
    np.add.at(h2_bins, _X2Y, p)
    hm2 = _entropy(h2_bins)
    
    mmax = max(hm0, hm1, hm2)
    if mmax <= 1e-12:
        return 0.0
    return hd / mmax


def _build_from_theta(theta):
    """Build distribution P from log-weights theta."""
    # theta is a 10x10 array of log-weights
    w = np.exp(theta - np.max(theta))  # numerical stability
    s = w.sum()
    if s <= 1e-12:
        return np.full((_N, _N), 1.0 / (_N * _N))
    return w / s


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    The baseline is the uniform distribution over the grid.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    t0 = time.time()
    DEADLINE = 595.0  # safety margin before 600s limit

    # Start with uniform distribution
    P_uniform = np.ones((_N, _N), dtype=np.float64) / (_N * _N)
    best_P = P_uniform.copy()
    best_score = _compute_value(P_uniform)

    def guard():
        return time.time() - t0 < DEADLINE

    # Strategy 1: Try structured distributions based on linear priors
    # theta[y, x] = a*y + b*x + c
    Y_grid, X_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing='ij')
    
    # Try various linear priors
    a_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4]
    b_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4]
    
    for a in a_vals:
        if not guard():
            break
        for b in b_vals:
            if not guard():
                break
            theta = a * Y_grid + b * X_grid
            P = _build_from_theta(theta)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()

    # Strategy 2: Local optimization around best found so far
    # Use L-BFGS-B to optimize the log-weights directly
    if guard():
        # Convert best_P to log-weights
        P_safe = np.maximum(best_P, 1e-15)
        theta0 = np.log(P_safe)
        
        def objective(theta):
            P = _build_from_theta(theta)
            return -_compute_value(P)
        
        try:
            res = minimize(objective, theta0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.x is not None and guard():
                P_opt = _build_from_theta(res.x)
                score_opt = _compute_value(P_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_P = P_opt.copy()
        except Exception:
            pass

    # Strategy 3: Try distributions concentrated on specific patterns
    # e.g., diagonal, anti-diagonal, or other structured patterns
    if guard():
        # Try distributions with varying concentration on diagonals
        for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
            if not guard():
                break
            # Weight based on |x - y|
            dist = np.abs(X_grid - Y_grid)
            theta = -alpha * dist
            P = _build_from_theta(theta)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()

    # Final cleanup: ensure valid output
    P = np.clip(best_P, 0.0, None)
    s = P.sum()
    if s <= 1e-12:
        P = P_uniform.copy()
    else:
        P = P / s
    
    return P

# EVOLVE-BLOCK-END
