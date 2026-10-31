"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize, differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (TARGET_SUM / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = TARGET_SUM - h.sum()
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _binomial_cdf(k, p=0.5):
    """Return the CDF of Binomial(k, p) evaluated at 0..k."""
    # Using math.comb for small k
    probs = np.zeros(k+1)
    for i in range(k+1):
        probs[i] = math.comb(k, i) * (p**i) * ((1-p)**(k-i))
    cdf = np.cumsum(probs)
    # Normalize to 0..1
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        cdf = np.linspace(0, 1, k+1)
    return cdf

def _generate_h(A, B, W, k, pos):
    """
    Generate h with a single transition window A->B->A? 
    Or just A->B? 
    Let's try A->B->A to allow return to A.
    Structure:
    - A from 0 to pos
    - Transition A->B from pos to pos+W
    - B from pos+W to pos+2W (if we want symmetry? No, just B plateau)
    - Transition B->A from pos+2W to pos+3W
    - A from pos+3W to N
    
    Actually, let's keep it simple:
    - A from 0 to pos
    - Transition A->B from pos to pos+W
    - B from pos+W to N-W
    - Transition B->A from N-W to N
    """
    h = np.zeros(N)
    
    # Plateau A
    h[:pos] = A
    
    # Transition A->B
    # Use binomial CDF for smoothness
    if k > 0:
        cdf = _binomial_cdf(k)
        # Interpolate cdf to W points
        # cdf is defined on 0..k. We map 0..W to 0..k
        x_interp = np.linspace(0, k, W)
        y_interp = np.interp(x_interp, np.arange(k+1), cdf)
        h[pos:pos+W] = A + (B - A) * y_interp
    else:
        h[pos:pos+W] = A + (B - A) * np.linspace(0, 1, W)
        
    # Plateau B
    if N - W > pos + W:
        h[pos+W:N-W] = B
        
    # Transition B->A
    if k > 0:
        cdf = _binomial_cdf(k)
        x_interp = np.linspace(0, k, W)
        y_interp = np.interp(x_interp, np.arange(k+1), cdf)
        # Reverse for B->A
        h[N-W:N] = B - (B - A) * y_interp
    else:
        h[N-W:N] = B - (B - A) * np.linspace(0, 1, W)
        
    return h

def _param_objective(params):
    A, B, W, k, pos = params
    A = np.clip(A, 0, 1)
    B = np.clip(B, 0, 1)
    W = int(np.clip(W, 1, N//2))
    k = int(np.clip(k, 0, 20))
    pos = int(np.clip(pos, 0, N - 2*W))
    
    h = _generate_h(A, B, W, k, pos)
    h = _project_to_feasible(h)
    return _score(h)

def _construct():
    deadline = time.time() + 140.0
    
    # Initial best from analytic 2-block solution
    A_init = (3.0 - math.sqrt(5.0)) / 2.0
    B_init = 1.0 - A_init
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A_init
    h0[half:] = B_init
    h_final = h0.copy()
    current_score = _score(h0)
    
    # Search over parametric family
    try:
        if time.time() < deadline - 10.0:
            # A, B in [0,1], W in [1, 50], k in [0, 20], pos in [0, N-2W]
            # We need to handle bounds for differential_evolution
            # W and k are integers, but DE works on floats. We will round.
            
            bounds = [
                (0.0, 1.0),  # A
                (0.0, 1.0),  # B
                (1.0, 50.0), # W
                (0.0, 20.0), # k
                (0.0, 50.0)  # pos
            ]
            
            # Initial guess
            x0 = [A_init, B_init, 10.0, 5.0, 10.0]
            
            result = differential_evolution(
                _param_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-12,
                polish=True
            )
            
            if result.fun < current_score:
                A, B, W, k, pos = result.x
                W = int(np.clip(W, 1, N//2))
                k = int(np.clip(k, 0, 20))
                pos = int(np.clip(pos, 0, N - 2*W))
                
                h_new = _generate_h(A, B, W, k, pos)
                h_new = _project_to_feasible(h_new)
                score_new = _score(h_new)
                
                if score_new < current_score:
                    h_final = h_new.copy()
                    current_score = score_new
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
        
    return h_final

def construct():
    return _construct()
