"""Optimize 50 nonnegative step heights on [-1/4,1/4] to maximize
   (sum h_i^2) / (sum h_i * sup_k (h*h)_k)
   where (h*h)_k = sum_i h_i h_{i+k} is the autoconvolution at lag k.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50

def _autoconv_max(h):
    """Compute max_k sum_i h[i] * h[i-k] for the given height array."""
    c = np.correlate(h, h, mode='full')
    return c.max()

def _ratio(h):
    """Compute the ratio (sum h^2) / (sum h * max autoconv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    ac_max = _autoconv_max(h)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def _neg_ratio(h):
    return -_ratio(h)

def _coordinate_descent(h, max_iter=500, tol=1e-10):
    """Coordinate descent: optimize one coordinate at a time."""
    h = np.asarray(h, dtype=float).copy()
    n = len(h)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            h_old = h[i]
            # Try a few values around current
            candidates = [0.0]
            if h_old > 0:
                candidates.extend([h_old * 0.5, h_old * 1.5, h_old * 0.1, h_old * 5.0])
            else:
                candidates.extend([0.1, 0.5, 1.0])
            # Add some random perturbations
            for _ in range(3):
                candidates.append(np.random.exponential(scale=max(h_old, 0.1)))
            
            best_val = _ratio(h)
            best_h_i = h_old
            for v in candidates:
                h[i] = v
                r = _ratio(h)
                if r > best_val:
                    best_val = r
                    best_h_i = v
            h[i] = best_h_i
            if abs(best_h_i - h_old) > tol:
                improved = True
        if not improved:
            break
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    rng = np.random.default_rng(42)
    
    # Strategy 1: Try very sparse signals (single spike, few spikes)
    # Single spike should give ratio ~1.0
    for pos in range(N):
        h = np.zeros(N)
        h[pos] = 1.0
        r = _ratio(h)
        if r > best_score:
            best_score = r
            best_h = h.copy()
    
    # Two spikes at various distances
    for i in range(N):
        for j in range(i+1, N):
            if time.time() > deadline - 60:
                break
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            r = _ratio(h)
            if r > best_score:
                best_score = r
                best_h = h.copy()
    
    # Strategy 2: Coordinate descent from diverse starts
    starts = []
    # Sparse random
    for _ in range(20):
        h = np.zeros(N)
        k = rng.integers(1, 8)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.uniform(0.5, 2.0, size=k)
        starts.append(h)
    # Dense random
    for _ in range(10):
        starts.append(rng.exponential(scale=1.0, size=N))
    # Structured
    x = np.linspace(0, N-1, N)
    for sigma in [2, 4, 6]:
        starts.append(np.exp(-0.5 * ((x - N//2) / sigma)**2))
    # Alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(alt)
    # Block
    for blk in [5, 10, 25]:
        h = np.zeros(N)
        h[:blk] = 1.0
        starts.append(h)
    
    for h0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            h_cd = _coordinate_descent(h0, max_iter=200)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    # Strategy 3: L-BFGS-B refinement from best
    if time.time() < deadline - 5:
        try:
            h0 = best_h.copy()
            if h0.max() > 0:
                h0 = h0 / h0.max()
            res = minimize(_neg_ratio, h0, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _neg_ratio(best_h):
                best_h = res.x
                best_score = -res.fun
        except Exception:
            pass
    
    # Strategy 4: More coordinate descent rounds from current best
    if time.time() < deadline - 3:
        try:
            h_cd = _coordinate_descent(best_h, max_iter=300)
            r = _ratio(h_cd)
            if r > best_score:
                best_score = r
                best_h = h_cd
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
