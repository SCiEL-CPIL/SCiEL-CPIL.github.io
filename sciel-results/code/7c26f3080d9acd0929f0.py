# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _shannon_bits(p):
    """Shannon entropy in bits, ignoring zero probabilities."""
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _proj_entropy(P, r):
    """Entropy of projection x + r*y."""
    n = P.shape[0]
    y = np.arange(n)
    x = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    vals = X + r * Y
    # Pool by value
    unique_vals = np.unique(vals)
    dist = np.zeros(len(unique_vals))
    for i, v in enumerate(unique_vals):
        dist[i] = P[vals == v].sum()
    return _shannon_bits(dist)

def _objective_flat(params):
    """Objective to minimize: -D/M (we want to maximize D/M)."""
    P = params.reshape(10, 10)
    # Clip negatives
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return 1e10
    P = P / s
    D = _proj_entropy(P, -1.0)
    H0 = _proj_entropy(P, 0.0)
    H1 = _proj_entropy(P, 1.0)
    H2 = _proj_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 1e10
    return -(D / M)

def _ratio(P):
    """Compute D/M ratio."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return 0.0
    P = P / s
    D = _proj_entropy(P, -1.0)
    H0 = _proj_entropy(P, 0.0)
    H1 = _proj_entropy(P, 1.0)
    H2 = _proj_entropy(P, 2.0)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s < 1e-15:
        return np.ones((10, 10)) / 100.0
    return P / s

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the entropy projection ratio H(x-y) / max(H(x), H(x+y), H(x+2y)).
    Returns a 10x10 array P where P[y, x] is the joint probability.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin
    
    # Start with uniform
    P_best = np.ones((10, 10), dtype=np.float64) / 100.0
    best_ratio = _ratio(P_best)
    
    # Try some analytical constructions
    candidates = [P_best.copy()]
    
    # Diagonal concentration: put mass on x-y = constant (maximizes D to log2(19))
    # but this makes x, x+y, x+2y concentrated
    for d in range(-9, 10):
        P = np.zeros((10, 10))
        for y in range(10):
            x = y + d
            if 0 <= x < 10:
                P[y, x] = 1.0
        s = P.sum()
        if s > 0:
            P = P / s
            candidates.append(P.copy())
    
    # Anti-diagonal patterns
    for c in range(19):
        P = np.zeros((10, 10))
        for y in range(10):
            x = c - y
            if 0 <= x < 10:
                P[y, x] = 1.0
        s = P.sum()
        if s > 0:
            P = P / s
            candidates.append(P.copy())
    
    # Corner-heavy distributions
    P = np.zeros((10, 10))
    P[0, 0] = 0.25
    P[0, 9] = 0.25
    P[9, 0] = 0.25
    P[9, 9] = 0.25
    candidates.append(P.copy())
    
    # Sparse random patterns
    np.random.seed(42)
    for _ in range(20):
        P = np.zeros((10, 10))
        # Pick k random cells
        k = np.random.randint(3, 15)
        idx = np.random.choice(100, size=k, replace=False)
        P[idx // 10, idx % 10] = 1.0
        candidates.append(P.copy())
    
    # Evaluate all candidates
    for P in candidates:
        r = _ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()
    
    # Local optimization from best candidate
    if time.time() < deadline - 120:
        x0 = P_best.flatten()
        try:
            res = minimize(_objective_flat, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            P_opt = _normalize(res.x.reshape(10, 10))
            r = _ratio(P_opt)
            if r > best_ratio:
                best_ratio = r
                P_best = P_opt.copy()
        except Exception:
            pass
    
    # Try differential evolution from a few starts if time permits
    if time.time() < deadline - 180:
        for seed in range(3):
            if time.time() >= deadline - 60:
                break
            np.random.seed(seed)
            try:
                res = differential_evolution(_objective_flat, 
                                             bounds=[(0, 1)] * 100,
                                             seed=seed,
                                             maxiter=50,
                                             tol=1e-8,
                                             popsize=15,
                                             init='latinhypercube',
                                             polish=True)
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
