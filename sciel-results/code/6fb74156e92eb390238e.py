# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize
from scipy.special import comb


def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using sorting.
    
    P = sum_{k,l} p_k * p_l * F(2*s_l - s_k)
    where F(t) = P[X1 + X2 < t]
    
    We precompute all pair sums and their probabilities, sort them,
    then for each (k,l) threshold, do a binary search to find the
    cumulative probability.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    # Compute all pair sums and their probabilities
    pair_sums = s[:, None] + s[None, :]  # shape (n, n)
    pair_probs = p[:, None] * p[None, :]  # shape (n, n)
    
    # Flatten and sort by pair sum
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    
    # Cumulative probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k, l), threshold = 2*s_l - s_k
    # We need F(threshold) = sum of pair_probs where pair_sum < threshold
    # Use searchsorted to find the index
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Find number of pair sums strictly less than threshold
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                prob = cum_probs[idx - 1]
            else:
                prob = 0.0
            P += p[k] * p[l] * prob
    return P


def _compute_P_brute(support, weights):
    """Brute force computation of P for verification."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = pair_sums < threshold
            P += p[k] * p[l] * pair_probs[mask].sum()
    return P


def _two_point_measure(a, p):
    """
    Two-point measure: {0, a} with probability p at 0 and (1-p) at a.
    Returns P[X1+X2+X3 < 2*X4].
    
    X1, X2, X3, X4 are i.i.d. with P(X=0) = p, P(X=a) = 1-p.
    
    Let q = 1-p.
    
    X1+X2+X3 can be: 0 (prob p^3), a (prob 3p^2 q), 2a (prob 3p q^2), 3a (prob q^3)
    2*X4 can be: 0 (prob p), 2a (prob q)
    
    P[X1+X2+X3 < 2*X4]:
    - If X4=0 (prob p): need X1+X2+X3 < 0, impossible. Contribution: 0
    - If X4=a (prob q): need X1+X2+X3 < 2a
      - X1+X2+X3 = 0: prob p^3, 0 < 2a yes (if a>0)
      - X1+X2+X3 = a: prob 3p^2 q, a < 2a yes (if a>0)
      - X1+X2+X3 = 2a: prob 3p q^2, 2a < 2a no (strict)
      - X1+X2+X3 = 3a: prob q^3, 3a < 2a no
    
    So P = q * (p^3 + 3p^2 q) = q * p^2 * (p + 3q) = q * p^2 * (p + 3(1-p)) = q * p^2 * (3 - 2p)
    
    Let's verify: P = (1-p) * p^2 * (3 - 2p)
    """
    if a <= 0:
        return 0.0
    q = 1.0 - p
    return q * p * p * (3.0 - 2.0 * p)


def _optimize_two_point():
    """Optimize the two-point measure {0, a} with probability p at 0."""
    # P = (1-p) * p^2 * (3 - 2p)
    # dP/dp = -p^2(3-2p) + (1-p)*2p*(3-2p) + (1-p)*p^2*(-2)
    # Let's just do numerical optimization
    
    def neg_P(p):
        return -_two_point_measure(1.0, p)
    
    result = minimize_scalar(neg_P, bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    p_opt = result.x
    P_opt = -result.fun
    return p_opt, P_opt


def _three_point_measure(params):
    """
    Three-point measure: {0, a, b} with probabilities p0, pa, pb.
    """
    p0, pa, pb, a, b = params
    if p0 < 0 or pa < 0 or pb < 0:
        return 1e10
    if p0 + pa + pb - 1.0 > 1e-6:
        return 1e10
    if a < 0 or b < 0:
        return 1e10
    if a > b:
        return 1e10
    
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([p0, pa, pb], dtype=np.float64)
    total = weights.sum()
    if total <= 1e-9:
        return 1e10
    
    p = weights / total
    n = 3
    
    # Precompute pair sums and probs
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cum_probs = np.cumsum(sorted_probs)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                prob = cum_probs[idx - 1]
            else:
                prob = 0.0
            P += p[k] * p[l] * prob
    return -P


def _optimize_three_point():
    """Optimize three-point measure {0, a, b}."""
    # Start from two-point optimum and add a third point
    p_opt, _ = _optimize_two_point()
    
    # Try various initializations
    best_P = -1e10
    best_params = None
    
    initial_guesses = [
        [p_opt * 0.9, p_opt * 0.1, 1.0 - p_opt, 0.5, 2.0],
        [p_opt * 0.8, p_opt * 0.2, 1.0 - p_opt, 0.5, 3.0],
        [p_opt * 0.7, p_opt * 0.3, 1.0 - p_opt, 0.5, 5.0],
        [0.5, 0.2, 0.3, 0.5, 2.0],
        [0.4, 0.3, 0.3, 0.5, 3.0],
        [0.3, 0.3, 0.4, 0.5, 5.0],
    ]
    
    for guess in initial_guesses:
        try:
            result = minimize(
                _three_point_measure,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x
        except Exception:
            pass
    
    return best_params, best_P


def _four_point_measure(params):
    """
    Four-point measure: {0, a, b, c} with probabilities p0, pa, pb, pc.
    """
    p0, pa, pb, pc, a, b, c = params
    if any(x < -1e-9 for x in [p0, pa, pb, pc, a, b, c]):
        return 1e10
    p0, pa, pb, pc = max(0, p0), max(0, pa), max(0, pb), max(0, pc)
    a, b, c = max(0, a), max(0, b), max(0, c)
    if a > b or b > c:
        return 1e10
    
    support = np.array([0.0, a, b, c], dtype=np.float64)
    weights = np.array([p0, pa, pb, pc], dtype=np.float64)
    total = weights.sum()
    if total <= 1e-9:
        return 1e10
    
    p = weights / total
    n = 4
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cum_probs = np.cumsum(sorted_probs)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            idx = np.searchsorted(sorted_sums, threshold, side='left')
            if idx > 0:
                prob = cum_probs[idx - 1]
            else:
                prob = 0.0
            P += p[k] * p[l] * prob
    return -P


def _optimize_four_point(start_params=None):
    """Optimize four-point measure {0, a, b, c}."""
    best_P = -1e10
    best_params = None
    
    if start_params is None:
        start_params = []
    
    # Add some initial guesses
    initial_guesses = [
        [0.4, 0.3, 0.2, 0.1, 0.5, 2.0, 5.0],
        [0.3, 0.3, 0.25, 0.15, 0.5, 3.0, 8.0],
        [0.35, 0.25, 0.25, 0.15, 0.5, 2.0, 4.0],
        [0.3, 0.3, 0.2, 0.2, 0.5, 2.5, 6.0],
    ]
    
    if start_params:
        initial_guesses.append(start_params)
    
    for guess in initial_guesses:
        try:
            result = minimize(
                _four_point_measure,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            P_val = -result.fun
            if P_val > best_P:
                best_P = P_val
                best_params = result.x
        except Exception:
            pass
    
    return best_params, best_P


def _geometric_measure(r, n_points):
    """
    Geometric-like measure: support at {0, r, r^2, ..., r^(n-1)} with
    weights proportional to (1-r)*r^i or similar.
    """
    if r <= 0 or r >= 1:
        return None, 0.0
    
    support = np.array([r**i for i in range(n_points)], dtype=np.float64)
    weights = np.array([(1-r) * r**i for i in range(n_points)], dtype=np.float64)
    
    P = _compute_P_fast(support, weights)
    return (support, weights), P


def _optimize_geometric():
    """Optimize geometric measure."""
    best_P = 0.0
    best_result = None
    
    for n_points in [3, 4, 5, 6, 8]:
        def neg_P(r):
            if r <= 0 or r >= 1:
                return 1e10
            support = np.array([r**i for i in range(n_points)], dtype=np.float64)
            weights = np.array([(1-r) * r**i for i in range(n_points)], dtype=np.float64)
            P = _compute_P_fast(support, weights)
            return -P
        
        # Try a grid of r values
        for r in np.linspace(0.1, 0.95, 20):
            P = -neg_P(r)
            if P > best_P:
                best_P = P
                best_result = (support, weights)
    
    return best_result, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Strategy 1: Two-point measure {0, a}
    p_opt, P_two = _optimize_two_point()
    if P_two > best_P:
        best_P = P_two
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 2: Three-point measure {0, a, b}
    try:
        params3, P_three = _optimize_three_point()
        if params3 is not None and P_three > best_P:
            best_P = P_three
            p0, pa, pb, a, b = params3
            p0, pa, pb = max(0, p0), max(0, pa), max(0, pb)
            total = p0 + pa + pb
            if total > 1e-9:
                best_support = np.array([0.0, a, b], dtype=np.float64)
                best_weights = np.array([p0, pa, pb], dtype=np.float64)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 3: Four-point measure {0, a, b, c}
    try:
        # Use the three-point result as a starting point if available
        start4 = None
        if params3 is not None:
            p0, pa, pb, a, b = params3
            start4 = [p0 * 0.8, pa * 0.8, pb * 0.8, 0.2 * (p0 + pa + pb), 
                      a * 0.8, b * 0.8, b * 1.5]
        
        params4, P_four = _optimize_four_point(start4)
        if params4 is not None and P_four > best_P:
            best_P = P_four
            p0, pa, pb, pc, a, b, c = params4
            p0, pa, pb, pc = max(0, p0), max(0, pa), max(0, pb), max(0, pc)
            total = p0 + pa + pb + pc
            if total > 1e-9:
                best_support = np.array([0.0, a, b, c], dtype=np.float64)
                best_weights = np.array([p0, pa, pb, pc], dtype=np.float64)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_support, best_weights
    
    # Strategy 4: Geometric measure
    try:
        geom_result, P_geom = _optimize_geometric()
        if geom_result is not None and P_geom > best_P:
            best_P = P_geom
            best_support, best_weights = geom_result
    except Exception:
        pass
    
    # Validate final result
    if best_support is None or best_weights is None:
        # Fallback to baseline
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Ensure validity
        best_support = np.asarray(best_support, dtype=np.float64).flatten()
        best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
        
        # Remove any invalid entries
        mask = (best_support >= -1e-9) & (best_weights >= -1e-9) & \
               np.isfinite(best_support) & np.isfinite(best_weights)
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        
        # Clip small negative values
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        
        if len(best_support) == 0 or best_weights.sum() <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
