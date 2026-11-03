import numpy as np
from scipy.optimize import minimize
import time

N = 50
MAX_TIME = 115.0  # safety margin before 120s kill

def _autoconvolve(h):
    """Compute autoconvolution of h (full length 2N-1)."""
    return np.convolve(h, h, mode='full')

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv(h)||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    if l1 <= 1e-300:
        return 0.0
    ac = _autoconvolve(h)
    sup = np.max(np.abs(ac))
    if sup <= 1e-300:
        return 0.0
    return l2_sq / (l1 * sup)

def _objective(x):
    """Negative score to minimize."""
    h = np.maximum(x, 0.0)  # enforce nonnegativity via projection
    return -_score(h)

def _grad(x):
    """Gradient approximation via finite differences (used only if needed)."""
    return None  # Let scipy use its default

def _refine(h0, max_iters=2000, tol=1e-10, time_budget=30.0):
    """Local refinement using L-BFGS-B with nonnegativity constraint."""
    h0 = np.maximum(np.asarray(h0, dtype=np.float64), 0.0)
    best_h = h0.copy()
    best_score = _score(h0)
    start = time.time()
    
    # Normalize to unit L1 for better conditioning
    l1 = np.sum(h0)
    if l1 < 1e-300:
        return best_h, best_score
    h0 = h0 / l1
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    for attempt in range(3):
        if time.time() - start > time_budget:
            break
        try:
            res = minimize(
                _objective,
                h0.copy(),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': max_iters, 'ftol': tol, 'gtol': tol, 'disp': False}
            )
            h_new = np.maximum(res.x, 0.0)
            s_new = _score(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new.copy()
                h0 = h_new.copy()
        except Exception:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    
    # Start with several diverse initial guesses
    candidates = []
    
    # 1. All ones (the seed)
    candidates.append(np.ones(N))
    
    # 2. Uniform
    candidates.append(np.ones(N) / N)
    
    # 3. Triangular
    tri = np.zeros(N)
    peak = N // 2
    for i in range(N):
        tri[i] = 1.0 - abs(i - peak) / (peak + 1)
    candidates.append(tri)
    
    # 4. Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-2 * x**2)
    candidates.append(gauss)
    
    # 5. Edge-peaked
    edge = np.zeros(N)
    edge[0] = 1.0
    edge[-1] = 1.0
    candidates.append(edge)
    
    # 6. Sparse center
    sparse = np.zeros(N)
    sparse[N//2] = 1.0
    candidates.append(sparse)
    
    # 7. Alternating high-low
    alt = np.zeros(N)
    alt[::2] = 1.0
    candidates.append(alt)
    
    # 8. Power-law decay from center
    pw = np.zeros(N)
    for i in range(N):
        d = abs(i - N // 2)
        pw[i] = 1.0 / (1.0 + d)
    candidates.append(pw)
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        r = rng.random(N)
        r = r ** rng.uniform(0.5, 3.0)  # various shapes
        candidates.append(r)
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Refine each candidate
    for i, h0 in enumerate(candidates):
        if time.time() - start_time > MAX_TIME - 5:
            break
        time_budget = min(15.0, MAX_TIME - 5 - (time.time() - start_time))
        if time_budget < 2:
            break
        try:
            h_ref, s_ref = _refine(h0, time_budget=time_budget)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref.copy()
        except Exception:
            pass
    
    # Final high-precision polish on the best candidate
    if time.time() - start_time < MAX_TIME - 3:
        remaining = MAX_TIME - 3 - (time.time() - start_time)
        if remaining > 2:
            try:
                h_final, s_final = _refine(
                    best_h, 
                    max_iters=5000, 
                    tol=1e-14, 
                    time_budget=remaining
                )
                if s_final > best_score:
                    best_score = s_final
                    best_h = h_final.copy()
            except Exception:
                pass
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0.0)
    
    # Convert to list of floats
    return [float(x) for x in best_h]
