# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

DEADLINE = 620  # seconds, safety margin before 630s kill

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given measure."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0
    
    # Pairwise sums
    pair_sums = s[:, None] + s[None, :]  # shape (n, n)
    pair_probs = p[:, None] * p[None, :]  # shape (n, n)
    
    # P = sum_{k,l} p_k * p_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} p_i*p_j)
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            mask = pair_sums < threshold
            P += p[k] * p[l] * pair_probs[mask].sum()
    return P


def _compute_P_fast(support, weights):
    """Vectorized computation of P[X1+X2+X3 < 2*X4]."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0
    
    # For each (k, l), threshold is 2*s_l - s_k
    # We need sum over i,j where s_i + s_j < threshold
    # 
    # Approach: for each l, sort by s_i + s_j < 2*s_l - s_k for each k
    # This is still O(n^3) in worst case but with better constants
    
    # Alternative: precompute all pairwise sums sorted
    pair_sums = s[:, None] + s[None, :]  # (n, n)
    pair_probs = p[:, None] * p[None, :]  # (n, n)
    
    # Flatten
    ps_flat = pair_sums.ravel()
    pp_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    order = np.argsort(ps_flat, kind='mergesort')
    ps_sorted = ps_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.concatenate([[0.0], np.cumsum(pp_sorted)])
    
    # Now for each (k, l), threshold = 2*s_l - s_k
    # Find number of elements < threshold using searchsorted
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ps_sorted, threshold, side='left')
            P += p[k] * p[l] * pp_cumsum[idx]
    return P


def _optimize_two_point():
    """
    Two-point measure: support = {0, a} with probabilities (1-p, p).
    P[X1+X2+X3 < 2*X4]
    
    X4 = 0 with prob (1-p): 2*X4 = 0, need X1+X2+X3 < 0, impossible since all >= 0.
    X4 = a with prob p: 2*X4 = 2a, need X1+X2+X3 < 2a.
    
    X1+X2+X3 can be 0, a, 2a, 3a.
    - 0 < 2a: always true if a > 0
    - a < 2a: true if a > 0
    - 2a < 2a: false (strict inequality)
    - 3a < 2a: false
    
    So P[X1+X2+X3 < 2a] = P[sum < 2a] = P[sum <= a]
    = P[all three are 0] + P[exactly one is a]
    = (1-p)^3 + 3*p*(1-p)^2
    
    Overall P = p * [(1-p)^3 + 3*p*(1-p)^2]
    = p*(1-p)^2 * [(1-p) + 3p]
    = p*(1-p)^2 * (1 + 2p)
    
    Maximize f(p) = p*(1-p)^2*(1+2p)
    f'(p) = ...
    Let's just optimize numerically.
    """
    def neg_f(p):
        p = max(0.0, min(1.0, p))
        return -(p * (1-p)**2 * (1 + 2*p))
    
    best_p = 0.5
    best_val = -1.0
    for p0 in np.linspace(0.01, 0.99, 50):
        res = minimize(neg_f, [p0], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        p_opt = max(0.0, min(1.0, res.x[0]))
        val = -res.fun
        if val > best_val:
            best_val = val
            best_p = p_opt
    
    p_opt = best_p
    # Support: {0, a}. The value a doesn't matter as long as a > 0 (for the strict inequality structure)
    # Actually, let's verify: with a=1, support = [0, 1]
    support = np.array([0.0, 1.0], dtype=np.float64)
    weights = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    
    # Verify with full computation
    P_val = _compute_P(support, weights)
    return support, weights, P_val


def _optimize_three_point():
    """
    Three-point measure: support = {0, 1, a} with probabilities (p0, p1, p2).
    Optimize over p0, p1, p2, a.
    """
    def objective(params):
        p0 = max(0.0, params[0])
        p1 = max(0.0, params[1])
        p2 = max(0.0, params[2])
        a = max(0.01, params[3])
        
        total = p0 + p1 + p2
        if total <= 1e-9:
            return 0.0
        p0n = p0 / total
        p1n = p1 / total
        p2n = p2 / total
        
        support = np.array([0.0, 1.0, a], dtype=np.float64)
        weights = np.array([p0n, p1n, p2n], dtype=np.float64)
        
        P = _compute_P(support, weights)
        return -P
    
    best_params = None
    best_val = -1.0
    
    random.seed(42)
    np.random.seed(42)
    
    n_starts = 20
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        # Random starting point
        p0 = random.uniform(0.1, 0.8)
        p1 = random.uniform(0.05, 0.5)
        p2 = random.uniform(0.05, 0.5)
        a = random.uniform(0.5, 5.0)
        
        x0 = np.array([p0, p1, p2, a])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to two-point
        s, w, P = _optimize_two_point()
        return s, w, P
    
    p0 = max(0.0, best_params[0])
    p1 = max(0.0, best_params[1])
    p2 = max(0.0, best_params[2])
    a = max(0.01, best_params[3])
    
    total = p0 + p1 + p2
    p0n = p0 / total
    p1n = p1 / total
    p2n = p2 / total
    
    support = np.array([0.0, 1.0, a], dtype=np.float64)
    weights = np.array([p0n, p1n, p2n], dtype=np.float64)
    
    P_val = _compute_P(support, weights)
    return support, weights, P_val


def _optimize_four_point():
    """
    Four-point measure: support = {0, 1, a, b} with probabilities (p0, p1, p2, p3).
    """
    def objective(params):
        p0 = max(0.0, params[0])
        p1 = max(0.0, params[1])
        p2 = max(0.0, params[2])
        p3 = max(0.0, params[3])
        a = max(0.01, params[4])
        b = max(0.01, params[5])
        
        total = p0 + p1 + p2 + p3
        if total <= 1e-9:
            return 0.0
        p0n = p0 / total
        p1n = p1 / total
        p2n = p2 / total
        p3n = p3 / total
        
        support = np.array([0.0, 1.0, a, b], dtype=np.float64)
        weights = np.array([p0n, p1n, p2n, p3n], dtype=np.float64)
        
        P = _compute_P(support, weights)
        return -P
    
    best_params = None
    best_val = -1.0
    
    random.seed(123)
    np.random.seed(123)
    
    n_starts = 15
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        p0 = random.uniform(0.2, 0.7)
        p1 = random.uniform(0.05, 0.4)
        p2 = random.uniform(0.05, 0.4)
        p3 = random.uniform(0.01, 0.3)
        a = random.uniform(0.5, 3.0)
        b = random.uniform(2.0, 8.0)
        
        x0 = np.array([p0, p1, p2, p3, a, b])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        s, w, P = _optimize_three_point()
        return s, w, P
    
    p0 = max(0.0, best_params[0])
    p1 = max(0.0, best_params[1])
    p2 = max(0.0, best_params[2])
    p3 = max(0.0, best_params[3])
    a = max(0.01, best_params[4])
    b = max(0.01, best_params[5])
    
    total = p0 + p1 + p2 + p3
    p0n = p0 / total
    p1n = p1 / total
    p2n = p2 / total
    p3n = p3 / total
    
    support = np.array([0.0, 1.0, a, b], dtype=np.float64)
    weights = np.array([p0n, p1n, p2n, p3n], dtype=np.float64)
    
    P_val = _compute_P(support, weights)
    return support, weights, P_val


def _optimize_geometric():
    """
    Geometric-like measure: support = {r^i for i=0..n-1} with geometric weights.
    """
    def objective(params):
        r = max(0.1, params[0])
        alpha = max(0.1, params[1])
        n = int(params[2]) if params[2] >= 2 else 2
        n = min(n, 30)  # Cap to keep computation fast
        
        support = r ** np.arange(n, dtype=np.float64)
        weights = alpha ** np.arange(n, dtype=np.float64)
        
        total = weights.sum()
        if total <= 1e-9:
            return 0.0
        weights_n = weights / total
        
        P = _compute_P(support, weights_n)
        return -P
    
    best_params = None
    best_val = -1.0
    
    random.seed(777)
    
    n_starts = 10
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        r = random.uniform(0.5, 3.0)
        alpha = random.uniform(0.1, 0.9)
        n = random.randint(3, 20)
        
        x0 = np.array([r, alpha, float(n)])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return None, None, 0.0
    
    r = max(0.1, best_params[0])
    alpha = max(0.1, best_params[1])
    n = int(round(best_params[2]))
    n = max(2, min(n, 30))
    
    support = r ** np.arange(n, dtype=np.float64)
    weights = alpha ** np.arange(n, dtype=np.float64)
    total = weights.sum()
    weights_n = weights / total
    
    P_val = _compute_P(support, weights_n)
    return support, weights_n, P_val


def _optimize_mixed():
    """
    Mixed strategy: combine a small atomic mass at 0 with a continuous-like spread.
    Support: {0, epsilon, 1, 2, ..., m} with weights favoring 0 and large values.
    """
    def objective(params):
        p0 = max(0.01, params[0])  # Mass at 0
        m = int(params[1]) if params[1] >= 1 else 1
        m = min(m, 20)
        spread = max(0.1, params[2])
        
        # Remaining mass distributed over {1, 2, ..., m}
        remaining = 1.0 - p0
        if remaining <= 0:
            remaining = 0.01
            p0 = 0.99
        
        # Geometric weights over {1, ..., m}
        w_rest = spread ** np.arange(m, dtype=np.float64)
        w_rest_sum = w_rest.sum()
        if w_rest_sum <= 1e-9:
            w_rest = np.ones(m)
            w_rest_sum = m
        w_rest = w_rest / w_rest_sum * remaining
        
        support = np.concatenate([[0.0], np.arange(1, m+1, dtype=np.float64)])
        weights = np.concatenate([[p0], w_rest])
        
        total = weights.sum()
        if total <= 1e-9:
            return 0.0
        weights_n = weights / total
        
        P = _compute_P(support, weights_n)
        return -P
    
    best_params = None
    best_val = -1.0
    
    random.seed(999)
    
    n_starts = 12
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        p0 = random.uniform(0.5, 0.95)
        m = random.randint(3, 15)
        spread = random.uniform(0.3, 0.9)
        
        x0 = np.array([p0, float(m), spread])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return None, None, 0.0
    
    p0 = max(0.01, best_params[0])
    m = int(round(best_params[1]))
    m = max(1, min(m, 20))
    spread = max(0.1, best_params[2])
    
    remaining = 1.0 - p0
    if remaining <= 0:
        remaining = 0.01
        p0 = 0.99
    
    w_rest = spread ** np.arange(m, dtype=np.float64)
    w_rest_sum = w_rest.sum()
    if w_rest_sum <= 1e-9:
        w_rest = np.ones(m)
        w_rest_sum = m
    w_rest = w_rest / w_rest_sum * remaining
    
    support = np.concatenate([[0.0], np.arange(1, m+1, dtype=np.float64)])
    weights = np.concatenate([[p0], w_rest])
    
    total = weights.sum()
    weights_n = weights / total
    
    P_val = _compute_P(support, weights_n)
    return support, weights_n, P_val


def _optimize_large_spread():
    """
    Strategy: put most mass at 0 and a tiny mass at a very large value L.
    This makes X4 rarely large, but when it is, 2*X4 is huge.
    Actually, we want X4 to be large often enough to make 2*X4 large.
    
    Let's try: support = {0, L} with mass (1-p, p) where p is small but L is large.
    We already did two-point. Let's try three-point: {0, a, L} with L >> a.
    """
    def objective(params):
        p0 = max(0.0, params[0])
        p1 = max(0.0, params[1])
        p2 = max(0.0, params[2])
        a = max(0.01, params[3])
        L = max(1.0, params[4])
        
        total = p0 + p1 + p2
        if total <= 1e-9:
            return 0.0
        p0n = p0 / total
        p1n = p1 / total
        p2n = p2 / total
        
        support = np.array([0.0, a, L], dtype=np.float64)
        weights = np.array([p0n, p1n, p2n], dtype=np.float64)
        
        P = _compute_P(support, weights)
        return -P
    
    best_params = None
    best_val = -1.0
    
    random.seed(555)
    
    n_starts = 15
    for i in range(n_starts):
        if time.time() > DEADLINE:
            break
        p0 = random.uniform(0.3, 0.8)
        p1 = random.uniform(0.05, 0.5)
        p2 = random.uniform(0.05, 0.4)
        a = random.uniform(0.5, 2.0)
        L = random.uniform(5.0, 50.0)
        
        x0 = np.array([p0, p1, p2, a, L])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return None, None, 0.0
    
    p0 = max(0.0, best_params[0])
    p1 = max(0.0, best_params[1])
    p2 = max(0.0, best_params[2])
    a = max(0.01, best_params[3])
    L = max(1.0, best_params[4])
    
    total = p0 + p1 + p2
    p0n = p0 / total
    p1n = p1 / total
    p2n = p2 / total
    
    support = np.array([0.0, a, L], dtype=np.float64)
    weights = np.array([p0n, p1n, p2n], dtype=np.float64)
    
    P_val = _compute_P(support, weights)
    return support, weights, P_val


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    
    # Try multiple strategies and pick the best
    candidates = []
    
    # Strategy 1: Two-point (analytically optimal ~0.4725)
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_two_point()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Strategy 2: Three-point
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_three_point()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Strategy 3: Four-point
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_four_point()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Strategy 4: Geometric
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_geometric()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Strategy 5: Mixed
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_mixed()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Strategy 6: Large spread
    if time.time() - start_time < DEADLINE:
        s, w, P = _optimize_large_spread()
        if P > 0 and all(w >= -1e-9) and all(s >= -1e-9):
            candidates.append((P, s, w))
    
    # Baseline: uniform
    n = 32
    support_base = np.arange(n, dtype=np.float64)
    weights_base = np.ones(n, dtype=np.float64) / n
    P_base = _compute_P(support_base, weights_base)
    candidates.append((P_base, support_base, weights_base))
    
    # Pick the best
    if not candidates:
        return support_base, weights_base
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_P, best_s, best_w = candidates[0]
    
    # Final validation
    if len(best_s) != len(best_w) or len(best_s) < 1:
        return support_base, weights_base
    if np.any(~np.isfinite(best_s)) or np.any(~np.isfinite(best_w)):
        return support_base, weights_base
    if np.any(best_s < -1e-9) or np.any(best_w < -1e-9):
        return support_base, weights_base
    if best_w.sum() <= 1e-9:
        return support_base, weights_base
    
    return best_s, best_w


# EVOLVE-BLOCK-END
