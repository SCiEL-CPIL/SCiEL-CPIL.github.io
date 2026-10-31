"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _compute_overlap(h, shifts):
    """Compute overlap of h with shifted (1-h) for given shifts."""
    n = len(h)
    overlaps = np.zeros(len(shifts))
    for idx, k in enumerate(shifts):
        if k <= 0:
            # shift left by k: h[i] * (1-h[i+k]) for i in [0, n-k-1]
            if k < n:
                valid = slice(0, n - k)
                overlaps[idx] = np.sum(h[valid] * (1 - h[valid + k]))
        else:
            # shift right by k: h[i] * (1-h[i-k]) for i in [k, n-1]
            if k < n:
                valid = slice(k, n)
                overlaps[idx] = np.sum(h[valid] * (1 - h[valid - k]))
    return overlaps

def _max_overlap(h):
    """Compute maximum overlap over all positive shifts."""
    n = len(h)
    max_ov = 0.0
    for k in range(1, n):
        ov = np.sum(h[:n-k] * (1 - h[k:]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        overlaps[k-1] = np.sum(h[:n-k] * (1 - h[k:]))
    # Soft max via log-sum-exp
    m = np.max(overlaps)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    
    # Try a structured initialization: alternating pattern
    # h[i] = 0.5 + a * cos(2*pi*i*p/N) for some frequency p
    best_h = h0.copy()
    best_score = _max_overlap(h0)
    
    # Parameterize h as 0.5 + sum of harmonics, ensure bounds [0,1] and sum constraint
    # Use a small number of Fourier coefficients
    def h_from_params(params):
        """Generate h from Fourier parameters. params = [a1, b1, a2, b2, ..., aK, bK, offset]"""
        n_foursier = len(params) - 1
        K = n_foursier // 2
        h = np.full(N, params[-1])  # offset
        for k in range(1, K+1):
            a = params[2*(k-1)]
            b = params[2*(k-1)+1]
            h += a * np.cos(2*np.pi*k*np.arange(N)/N) + b * np.sin(2*np.pi*k*np.arange(N)/N)
        # Clip to [0,1]
        h = np.clip(h, 0.0, 1.0)
        # Enforce sum constraint: adjust offset
        target_sum = N / 2.0
        current_sum = np.sum(h)
        if current_sum > 0:
            # Scale and re-clip
            h = h * (target_sum / current_sum)
            h = np.clip(h, 0.0, 1.0)
            # Re-adjust
            current_sum2 = np.sum(h)
            if abs(current_sum2 - target_sum) > 1e-10:
                h = h + (target_sum - current_sum2) / N
                h = np.clip(h, 0.0, 1.0)
        return h
    
    # Objective: minimize soft max overlap
    def objective(params):
        h = h_from_params(params)
        # Check sum constraint
        if abs(np.sum(h) - N/2.0) > 1e-6:
            return 1e10
        return _soft_max_overlap(h, beta=30.0)
    
    # Global search with differential evolution on Fourier coefficients
    K = 5  # number of harmonics
    n_params = 2*K + 1
    bounds = [(-0.5, 0.5)] * 2*K + [(0.3, 0.7)]
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=100, popsize=15,
                tol=1e-6, seed=42,
                polish=True,
                updating='immediate'
            )
            h_candidate = h_from_params(result.x)
            if abs(np.sum(h_candidate) - N/2.0) < 1e-6:
                score = _max_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Local polish with scipy minimize
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            # Use the best h so far as initial guess, work in unconstrained space
            # Project back to feasible set
            h_init = best_h.copy()
            
            # Local optimization using projected gradient or direct minimization
            def obj_local(x):
                h = np.clip(x, 0.0, 1.0)
                # Enforce sum
                s = np.sum(h)
                if s > 0:
                    h = h * (N/2.0 / s)
                    h = np.clip(h, 0.0, 1.0)
                if abs(np.sum(h) - N/2.0) > 1e-8:
                    return 1e10
                return _soft_max_overlap(h, beta=50.0)
            
            res = minimize(obj_local, h_init, method='L-BFGS-B',
                          bounds=[(0,1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
            h_candidate = np.clip(res.x, 0.0, 1.0)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N/2.0 / s)
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            if abs(np.sum(h_candidate) - N/2.0) < 1e-6:
                score = _max_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            h_init = best_h.copy()
            def obj_polish(x):
                h = np.clip(x, 0.0, 1.0)
                s = np.sum(h)
                if s > 0:
                    h = h * (N/2.0 / s)
                    h = np.clip(h, 0.0, 1.0)
                if abs(np.sum(h) - N/2.0) > 1e-10:
                    return 1e10
                # Use actual max for precision
                return _max_overlap(h)
            
            res = minimize(obj_polish, h_init, method='L-BFGS-B',
                          bounds=[(0,1)]*N, options={'maxiter': 1000, 'ftol': 1e-14})
            h_candidate = np.clip(res.x, 0.0, 1.0)
            s = np.sum(h_candidate)
            if s > 0:
                h_candidate = h_candidate * (N/2.0 / s)
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
            if abs(np.sum(h_candidate) - N/2.0) < 1e-8:
                score = _max_overlap(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-10:
        best_h = best_h + (N/2.0 - s) / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-adjust
        s2 = np.sum(best_h)
        if abs(s2 - N/2.0) > 1e-10:
            best_h = best_h * (N/2.0 / s2)
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
