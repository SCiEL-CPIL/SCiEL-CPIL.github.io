# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 620.0  # Safety margin before 630s kill

def _shannon_bits(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_all_entropies(P):
    """
    Compute H_{-1}, H_0, H_1, H_2 for P[y, x].
    Returns (H_minus1, H_0, H_1, H_2).
    """
    n = P.shape[0]
    y = np.arange(n)
    x = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # x - y: values from -9 to 9
    diff = X - Y
    unique_diff = np.arange(-9, 10)
    dist_diff = np.array([P[diff == v].sum() for v in unique_diff])
    H_minus1 = _shannon_bits(dist_diff)
    
    # x: values from 0 to 9
    unique_x = np.arange(10)
    dist_x = np.array([P[:, v].sum() for v in unique_x])
    H_0 = _shannon_bits(dist_x)
    
    # x + y: values from 0 to 18
    sum_xy = X + Y
    unique_sum = np.arange(19)
    dist_sum = np.array([P[sum_xy == v].sum() for v in unique_sum])
    H_1 = _shannon_bits(dist_sum)
    
    # x + 2y: values from 0 to 27
    sum_x2y = X + 2 * Y
    unique_x2y = np.arange(28)
    dist_x2y = np.array([P[sum_x2y == v].sum() for v in unique_x2y])
    H_2 = _shannon_bits(dist_x2y)
    
    return H_minus1, H_0, H_1, H_2

def _ratio(P):
    """Compute D/M ratio."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return 0.0
    P = P / s
    D, H0, H1, H2 = _compute_all_entropies(P)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def _objective_flat(params):
    """Objective to minimize: -D/M (we want to maximize D/M)."""
    P = params.reshape(10, 10)
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return 1e10
    P = P / s
    D, H0, H1, H2 = _compute_all_entropies(P)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 1e10
    return -(D / M)

def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return np.ones((10, 10)) / 100.0
    return P / s

def _generate_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # Uniform
    candidates.append(np.ones((10, 10)) / 100.0)
    
    # Diagonal lines (x - y = constant)
    for d in range(-9, 10):
        P = np.zeros((10, 10))
        for y in range(10):
            x = y + d
            if 0 <= x < 10:
                P[y, x] = 1.0
        s = P.sum()
        if s > 0:
            candidates.append(P / s)
    
    # Anti-diagonal lines (x + y = constant)
    for c in range(19):
        P = np.zeros((10, 10))
        for y in range(10):
            x = c - y
            if 0 <= x < 10:
                P[y, x] = 1.0
        s = P.sum()
        if s > 0:
            candidates.append(P / s)
    
    # Corners
    P = np.zeros((10, 10))
    P[0, 0] = 1.0
    P[0, 9] = 1.0
    P[9, 0] = 1.0
    P[9, 9] = 1.0
    candidates.append(P / P.sum())
    
    # Edges
    for edge in ['top', 'bottom', 'left', 'right']:
        P = np.zeros((10, 10))
        if edge == 'top':
            P[0, :] = 1.0
        elif edge == 'bottom':
            P[9, :] = 1.0
        elif edge == 'left':
            P[:, 0] = 1.0
        elif edge == 'right':
            P[:, 9] = 1.0
        candidates.append(P / P.sum())
    
    # Random sparse patterns
    np.random.seed(42)
    for _ in range(30):
        P = np.zeros((10, 10))
        k = np.random.randint(3, 20)
        idx = np.random.choice(100, size=k, replace=False)
        P[idx // 10, idx % 10] = 1.0
        candidates.append(P / P.sum())
    
    # Checkerboard patterns
    for offset in [0, 1]:
        P = np.zeros((10, 10))
        for y in range(10):
            for x in range(10):
                if (x + y + offset) % 2 == 0:
                    P[y, x] = 1.0
        if P.sum() > 0:
            candidates.append(P / P.sum())
    
    # Concentrated in center
    P = np.zeros((10, 10))
    P[4:6, 4:6] = 1.0
    candidates.append(P / P.sum())
    
    # Cross pattern
    P = np.zeros((10, 10))
    P[5, :] = 1.0
    P[:, 5] = 1.0
    candidates.append(P / P.sum())
    
    return candidates

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the entropy projection ratio H(x-y) / max(H(x), H(x+y), H(x+2y)).
    Returns a 10x10 array P where P[y, x] is the joint probability.
    """
    start_time = time.time()
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    P_best = candidates[0].copy()
    best_ratio = _ratio(P_best)
    
    for P in candidates[1:]:
        r = _ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()
    
    remaining = _DEADLINE - time.time()
    
    # Local optimization from best candidate
    if remaining > 30:
        x0 = P_best.flatten()
        try:
            res = minimize(_objective_flat, x0, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
            P_opt = _normalize(res.x.reshape(10, 10))
            r = _ratio(P_opt)
            if r > best_ratio:
                best_ratio = r
                P_best = P_opt.copy()
        except Exception:
            pass
    
    # Try differential evolution if time permits
    remaining = _DEADLINE - time.time()
    if remaining > 60:
        for seed in range(5):
            if _DEADLINE - time.time() < 30:
                break
            np.random.seed(seed)
            try:
                res = differential_evolution(_objective_flat, 
                                             bounds=[(0, 1)] * 100,
                                             seed=seed,
                                             maxiter=100,
                                             tol=1e-10,
                                             popsize=20,
                                             init='latinhypercube',
                                             polish=True)
                P_opt = _normalize(res.x.reshape(10, 10))
                r = _ratio(P_opt)
                if r > best_ratio:
                    best_ratio = r
                    P_best = P_opt.copy()
            except Exception:
                pass
    
    # Final local polish
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        x0 = P_best.flatten()
        try:
            res = minimize(_objective_flat, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
            P_opt = _normalize(res.x.reshape(10, 10))
            r = _ratio(P_opt)
            if r > best_ratio:
                best_ratio = r
                P_best = P_opt.copy()
        except Exception:
            pass
    
    # Final cleanup
    P_best = _normalize(P_best)
    
    # Safety check
    if not np.all(np.isfinite(P_best)) or P_best.shape != (10, 10) or P_best.sum() < 1e-9:
        P_best = np.ones((10, 10)) / 100.0
    
    return P_best

# EVOLVE-BLOCK-END
