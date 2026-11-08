"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    h is a 1D array of step heights.
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    # Overlap at shift k is N/2 - A_k. We want min max overlap.
    return np.max(N/2 - A)

def _objective_soft(h, T=0.05):
    """
    Smooth surrogate for the max overlap.
    Uses log-sum-exp to approximate the max.
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    overlaps = N/2 - A
    # Soft max: T * log(sum(exp(o/T)))
    # To avoid overflow, subtract max
    m = np.max(overlaps)
    soft_max = m + T * np.log(np.sum(np.exp((overlaps - m) / T)))
    return soft_max

def _project(h):
    """
    Project h to [0, 1] and sum N/2.
    Simple iterative projection.
    """
    h = np.clip(h, 0, 1)
    for _ in range(10):
        diff = np.sum(h) - N/2
        if abs(diff) < 1e-9:
            break
        h -= diff / N
        h = np.clip(h, 0, 1)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Constant
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating
    alt = np.zeros(N)
    alt[0::2] = 0.25
    alt[1::2] = 0.75
    candidates.append(alt)
    
    # 3. Ramp
    ramp = np.linspace(0.2, 0.8, N)
    ramp = ramp - np.mean(ramp) + 0.5
    ramp = np.clip(ramp, 0, 1)
    diff = N/2 - np.sum(ramp)
    ramp += diff / N
    ramp = np.clip(ramp, 0, 1)
    candidates.append(ramp)
    
    # 4. Sine wave (low freq)
    sine1 = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N)
    sine1 = np.clip(sine1, 0, 1)
    diff = N/2 - np.sum(sine1)
    sine1 += diff / N
    sine1 = np.clip(sine1, 0, 1)
    candidates.append(sine1)
    
    # 5. Sine wave (high freq)
    sine2 = 0.5 + 0.3 * np.sin(2 * np.pi * 5 * np.arange(N) / N)
    sine2 = np.clip(sine2, 0, 1)
    diff = N/2 - np.sum(sine2)
    sine2 += diff / N
    sine2 = np.clip(sine2, 0, 1)
    candidates.append(sine2)
    
    # 6. Random
    rand = rng.uniform(0, 1, N)
    diff = N/2 - np.sum(rand)
    rand += diff / N
    rand = np.clip(rand, 0, 1)
    candidates.append(rand)
    
    # Evaluate initial candidates
    for c in candidates:
        score = _compute_max_overlap(c)
        if score < best_score:
            best_score = score
            best_h = c.copy()
    
    # Run L-BFGS-B from each candidate
    for i, c in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Define objective for optimizer
        # We optimize the soft objective, but check hard objective for best tracking
        def obj_wrapper(x):
            # Project to ensure constraints during optimization steps?
            # L-BFGS-B can handle bounds [0,1]. Sum constraint is linear equality, 
            # which L-BFGS-B doesn't handle directly. 
            # We can penalize sum deviation or project.
            # Projecting inside objective is non-smooth. 
            # Better: work in a parameterization that satisfies sum?
            # Or just use penalty.
            # Let's try projecting. It might hurt gradient accuracy but L-BFGS-B is robust.
            # Actually, let's just optimize the raw vector with bounds [0,1] and a penalty for sum.
            h = x
            sum_err = np.sum(h) - N/2
            penalty = 10.0 * sum_err**2
            soft_val = _objective_soft(h, T=0.05)
            return soft_val + penalty
        
        # Initial point
        x0 = c.copy()
        
        # Bounds
        bounds = [(0, 1) for _ in range(N)]
        
        try:
            res = minimize(
                obj_wrapper, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 2000}
            )
            
            # Evaluate result
            h_opt = res.x
            # Final projection to ensure strict validity
            h_opt = _project(h_opt)
            
            score = _compute_max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
        except Exception:
            pass
            
    # If we have time, do a second pass with tighter T or different starts?
    # Or just return.
    if time.time() < deadline - 5:
        # Try to refine best_h with a tighter soft-min
        def obj_refine(x):
            h = x
            sum_err = np.sum(h) - N/2
            penalty = 50.0 * sum_err**2
            soft_val = _objective_soft(h, T=0.01)
            return soft_val + penalty
            
        try:
            res2 = minimize(
                obj_refine, 
                best_h, 
                method='L-BFGS-B', 
                bounds=[(0, 1) for _ in range(N)],
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000}
            )
            h_refined = _project(res2.x)
            score_refined = _compute_max_overlap(h_refined)
            if score_refined < best_score:
                best_score = score_refined
                best_h = h_refined.copy()
        except Exception:
            pass

    return best_h

# EVOLVE-BLOCK-END
