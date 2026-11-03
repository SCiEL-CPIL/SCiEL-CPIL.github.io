"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _overlap_metric(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap for shift k: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the max over k of this overlap.
    """
    N = len(h)
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    
    # Compute all circular cross-correlations
    # overlap[k] = sum_i h[i] * (1-h)[(i+k) % N]
    # This is circular cross-correlation of h and (1-h)
    
    # Use FFT for efficiency
    h_padded = np.zeros(2 * N)
    omh_padded = np.zeros(2 * N)
    h_padded[:N] = h
    omh_padded[:N] = one_minus_h
    
    H = np.fft.fft(h_padded, n=2*N)
    OMH = np.fft.fft(omh_padded, n=2*N)
    
    # Circular cross-correlation
    # corr[k] = sum_i h[i] * omh[(i+k) % N]
    # This is the real part of ifft(conj(H) * OMH) shifted appropriately
    corr = np.real(np.fft.ifft(np.conj(H) * OMH))
    
    # The first N values of corr correspond to shifts 0..N-1
    # But we need to be careful with the indexing
    # Actually, let's just compute it directly for clarity and correctness
    # Since N=100, direct computation is fast enough
    
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_omh = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted_omh)
    
    return np.max(overlaps)

def _objective(params, N):
    """
    Objective function for optimization.
    params: vector of length N-1 (the last value is determined by sum constraint)
    """
    h = np.zeros(N)
    h[:N-1] = params
    h[N-1] = N/2.0 - np.sum(params)
    
    # Ensure non-negativity and upper bound
    # If any value is out of [0,1], return a large penalty
    if np.any(h < -1e-10) or np.any(h > 1.0 + 1e-10):
        return 1e6
    
    # Clamp to [0,1] for safety
    h = np.clip(h, 0.0, 1.0)
    
    return _overlap_metric(h)

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Strategy: Use differential evolution for global search, then polish with local optimization
    
    # Initial guess: alternating 0 and 1
    h_init = np.zeros(N)
    h_init[::2] = 1.0
    h_init[1::2] = 0.0
    
    # But this has sum = N/2, good. Let's check the overlap.
    # For shift 0: sum h[i]*(1-h[i]) = 0
    # For shift 1: sum h[i]*(1-h[i+1]) = sum of h[i]*1 for even i, h[i]*0 for odd i = N/2
    # Wait, let's think more carefully.
    
    # Actually, let's use a parameterization that ensures the sum constraint.
    # We'll optimize over N-1 free variables.
    
    def obj_constrained(x):
        h = np.zeros(N)
        h[:N-1] = x
        h[N-1] = N/2.0 - np.sum(x)
        if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
            return 1e6
        h = np.clip(h, 0.0, 1.0)
        return _overlap_metric(h)
    
    # Differential evolution
    bounds = [(0.0, 1.0)] * (N-1)
    
    best_result = None
    best_score = float('inf')
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            obj_constrained,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x.copy()
    except Exception as e:
        pass
    
    if time.time() > deadline - 20.0:
        # Not enough time for polishing, return what we have
        if best_result is not None:
            h = np.zeros(N)
            h[:N-1] = best_result
            h[N-1] = N/2.0 - np.sum(best_result)
            h = np.clip(h, 0.0, 1.0)
            # Renormalize to ensure sum is exactly N/2
            current_sum = np.sum(h)
            if current_sum > 0:
                h = h * (N/2.0) / current_sum
                h = np.clip(h, 0.0, 1.0)
            return h
        else:
            return np.full(N, 0.5)
    
    # Polish with SLSQP
    if best_result is not None:
        x0 = best_result
    else:
        # Fallback initial guess
        x0 = np.full(N-1, 0.5)
        x0[N-2] = N/2.0 - (N-1)*0.5  # This might be negative, so use a better init
    
    # Better initial guess for polishing
    x0 = np.zeros(N-1)
    x0[::2] = 1.0
    x0[1::2] = 0.0
    # Adjust to meet sum constraint
    current_sum = np.sum(x0)
    target = N/2.0 - 0.0  # Last element will be 0
    # Actually, let's just use the DE result if available
    
    if best_result is not None:
        x0 = best_result
    else:
        x0 = np.full(N-1, 0.5)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - (N/2.0 - 0.0)}  # This is wrong, last var is free
    ]
    
    # Actually, the constraint is that the last variable = N/2 - sum(x[:N-1])
    # And all variables in [0,1]
    
    try:
        slsqp_result = minimize(
            obj_constrained,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if slsqp_result.fun < best_score:
            best_score = slsqp_result.fun
            best_result = slsqp_result.x.copy()
    except Exception as e:
        pass
    
    # Construct final h
    if best_result is not None:
        h = np.zeros(N)
        h[:N-1] = best_result
        h[N-1] = N/2.0 - np.sum(best_result)
        h = np.clip(h, 0.0, 1.0)
        # Renormalize
        current_sum = np.sum(h)
        if current_sum > 1e-10:
            h = h * (N/2.0) / current_sum
            h = np.clip(h, 0.0, 1.0)
        return h
    else:
        return np.full(N, 0.5)

# EVOLVE-BLOCK-END
