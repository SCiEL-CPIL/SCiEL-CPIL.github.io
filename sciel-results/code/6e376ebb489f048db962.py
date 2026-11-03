import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h) * DELTA
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2) * DELTA
    # Autoconvolution: (f*f)(k*delta) = delta^2 * sum_i h_i h_{i+k} for lag k
    # For lag 0: delta * sum_i h_i^2
    # Compute all lags
    h_conv = np.convolve(h, h, mode='full')
    # h_conv[k] corresponds to lag (k - (N-1)) * delta
    # The value at lag 0 is h_conv[N-1] = sum_i h_i^2, and the actual integral is delta * that
    # For lag k != 0, the integral is delta^2 * h_conv[k]
    # So sup = max(delta * h_conv[N-1], delta^2 * max(h_conv[0:N-1], h_conv[N:]))
    lag0_val = DELTA * h_conv[N-1]
    other_lags = np.concatenate([h_conv[:N-1], h_conv[N:]])
    other_val = DELTA**2 * np.max(other_lags) if len(other_lags) > 0 else 0
    autoconv_max = max(lag0_val, other_val)
    if autoconv_max <= 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # The score is scale invariant, so we can work with any scaling.
    # Let's parameterize h directly and optimize.
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # Try various structured initializations
    x_idx = np.arange(N)
    
    # Sparse peaks - try different numbers of peaks
    for n_peaks in range(1, 20):
        if time.time() > deadline - 30:
            break
        # Try evenly spaced
        positions = np.linspace(2, N-3, n_peaks).astype(int)
        h_sparse = np.zeros(N)
        h_sparse[positions] = 1.0
        s = _score_raw(h_sparse)
        if s > best_score:
            best_score = s
            best_h = h_sparse.copy()
        
        # Try random positions
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            positions = np.random.choice(N, n_peaks, replace=False)
            h_sparse = np.zeros(N)
            h_sparse[positions] = 1.0
            s = _score_raw(h_sparse)
            if s > best_score:
                best_score = s
                best_h = h_sparse.copy()
    
    # Use differential evolution for global optimization
    # We maximize score, so minimize negative score
    def neg_score(h):
        return -_score_raw(h)
    
    bounds = [(0.0, 10.0)] * N
    
    # Run differential evolution from multiple starting points
    for restart in range(3):
        if time.time() > deadline - 20:
            break
        
        # Different bounds for diversity
        if restart == 0:
            b = [(0.0, 5.0)] * N
        elif restart == 1:
            b = [(0.0, 2.0)] * N
        else:
            b = [(0.0, 1.0)] * N
        
        try:
            res = differential_evolution(
                neg_score, b, 
                maxiter=100, popsize=15,
                tol=1e-6, mutation=(0.5, 1.5),
                recombination=0.9, seed=42+restart,
                polish=True
            )
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Local refinement with SLSQP
    def neg_obj(h):
        return -_score_raw(h)
    
    constraints = [{'type': 'ineq', 'fun': lambda h: h}]
    bounds = [(0.0, 50.0)] * N
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        
        # Perturb best solution
        h_start = best_h.copy()
        h_start += np.random.randn(N) * 0.1 * np.max(best_h)
        h_start = np.maximum(h_start, 0)
        
        try:
            res = minimize(neg_obj, h_start, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return list(best_h)
