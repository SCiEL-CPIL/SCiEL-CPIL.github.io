"""Optimized program for the second autocorrelation inequality task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 115  # seconds, safety margin before 120s kill

def _autoconv(h):
    """Compute autoconvolution of h with itself."""
    n = len(h)
    # Full autoconvolution has 2n-1 elements
    ac = np.zeros(2 * n - 1)
    for i in range(n):
        ac[i:i + n] += h[i] * h
    return ac

def _score(h):
    """Compute the score: (L2^2) / (L1 * sup|autoconv|)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    ac = _autoconv(h)
    sup_ac = np.max(np.abs(ac))
    if sup_ac <= 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _normalize_l1(h):
    """Normalize h to have L1 = 1."""
    s = np.sum(h)
    if s <= 0:
        return h
    return h / s

def _uniform():
    return np.ones(N) / N

def _sidon_like():
    """Create a distribution based on a Sidon set pattern."""
    # Use a quadratic residue-like pattern for low autocorrelation
    h = np.zeros(N)
    # Place mass at positions that form a good autocorrelation structure
    # Try a few known good patterns
    patterns = []
    
    # Pattern 1: evenly spaced
    for k in range(1, 8):
        pos = list(range(0, N, k))
        if len(pos) >= 3:
            patterns.append(pos)
    
    # Pattern 2: quadratic residues mod p
    for p in [47, 43, 41, 37, 31]:
        if p <= N:
            residues = [(i * i) % p for i in range(p)]
            unique_res = sorted(set(r for r in residues if r < N))
            if len(unique_res) >= 3:
                patterns.append(unique_res)
    
    # Pattern 3: random structured
    rng = np.random.RandomState(42)
    for trial in range(50):
        k = rng.randint(5, 15)
        pos = sorted(rng.choice(N, size=k, replace=False).tolist())
        patterns.append(pos)
    
    best_h = None
    best_score = 0
    for pos in patterns:
        h = np.zeros(N)
        for p in pos:
            h[p] = 1.0 / len(pos)
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    return best_h if best_h is not None else _uniform()

def _barker_like():
    """Try Barker-sequence-inspired patterns."""
    # Barker sequences have low sidelobes in autocorrelation
    # Map Barker-like ±1 patterns to nonnegative heights
    barker_seqs = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, -1, 1, 1, -1],
        [1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1],
    ]
    
    best_h = None
    best_score = 0
    for bs in barker_seqs:
        # Create a periodic extension or mapping
        for offset in range(min(N, 10)):
            h = np.zeros(N)
            for i in range(N):
                idx = (i + offset) % len(bs)
                h[i] = 1.0 if bs[idx] == 1 else 0.0
            if np.sum(h) > 0:
                h = _normalize_l1(h)
                s = _score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
    
    return best_h if best_h is not None else _uniform()

def _local_optimize(h0, max_time=30):
    """Local optimization using gradient-based methods."""
    h0 = _normalize_l1(h0)
    
    def neg_score(params):
        h = np.exp(params)  # ensures positivity
        h = h / np.sum(h)  # normalize L1
        return -_score(h)
    
    def neg_score_grad(params):
        h = np.exp(params)
        h = h / np.sum(h)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = neg_score(params)
        for i in range(len(params)):
            params_p = params.copy()
            params_p[i] += eps
            grad[i] = (neg_score(params_p) - f0) / eps
        return grad
    
    start_time = time.time()
    params0 = np.log(h0 + 1e-10)
    
    try:
        result = minimize(neg_score, params0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        elapsed = time.time() - start_time
        if elapsed < max_time:
            h_opt = np.exp(result.x)
            h_opt = h_opt / np.sum(h_opt)
            return h_opt
    except:
        pass
    
    return h0

def _random_search(num_trials=200, max_time=20):
    """Random search with local refinement."""
    rng = np.random.RandomState(123)
    best_h = _uniform()
    best_score = _score(best_h)
    
    start_time = time.time()
    for trial in range(num_trials):
        if time.time() - start_time > max_time:
            break
        
        # Generate random nonnegative vector
        h = rng.dirichlet(np.ones(N))
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    return best_h

def construct():
    start_time = time.time()
    
    # Generate candidate distributions
    candidates = []
    
    # Uniform baseline
    candidates.append(_uniform())
    
    # Sidon-like patterns
    candidates.append(_sidon_like())
    
    # Barker-like patterns
    candidates.append(_barker_like())
    
    # Random search
    if time.time() - start_time < TIME_LIMIT - 40:
        candidates.append(_random_search(num_trials=100, max_time=15))
    
    # Find best candidate
    best_h = None
    best_score = 0
    for h in candidates:
        h = _normalize_l1(h)
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization on best candidate
    if best_h is not None and time.time() - start_time < TIME_LIMIT - 15:
        h_opt = _local_optimize(best_h, max_time=min(25, TIME_LIMIT - (time.time() - start_time) - 5))
        s_opt = _score(h_opt)
        if s_opt > best_score:
            best_h = h_opt
            best_score = s_opt
    
    # Final safety check
    if best_h is None:
        best_h = _uniform()
    
    # Ensure nonnegative and valid
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    else:
        best_h = _uniform()
    
    return list(best_h)
