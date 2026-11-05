"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _hard_value(h):
    """Compute the actual evaluation value."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    # Use fftconvolve for speed if possible, but np.convolve is fine for N=400
    # fftconvolve requires power of 2 or similar for efficiency, but works generally.
    # Let's stick to np.convolve for simplicity and correctness, or use fftconvolve.
    # For N=400, np.convolve is ~160k ops. Very fast.
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(x, beta=100.0):
    """
    Objective to minimize: 2*N*soft_max(conv) / (sum(h))^2
    x is the parameter vector (delta) such that h = 1/N + x, sum(x)=0.
    """
    h = np.full(N, 1.0/N) + x
    # Ensure sum(h) is exactly 1? 
    # Due to floating point, sum(h) might drift.
    # We can re-normalize h to sum to 1, but that changes the gradient.
    # Better to just use the constraint in the optimization?
    # L-BFGS-B doesn't handle equality constraints well.
    # Let's just use h = x and add a penalty for sum(h) != 1?
    # Or just optimize h directly and divide by sum(h)^2.
    # The previous approach optimized h directly.
    # Let's stick to optimizing h directly to avoid constraint handling issues.
    
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    
    # Soft max
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    
    return 2.0 * N * log_sum_exp / (s * s)

def _soft_max_grad(x, beta=100.0):
    """
    Gradient of the soft-max objective.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    
    # Gradient of soft_max w.r.t. h
    # grad_soft_max[i] = sum_j weights[j] * (h[j-i] + h[i-j])
    # Using FFT for speed?
    # weights has length 2N-1.
    # We can compute the convolution of weights and h.
    # term1 = convolve(weights, h) ?
    # Let's use the direct O(N^2) loop for N=400, it's fast enough.
    grad_soft_max = np.zeros(N)
    for i in range(N):
        acc = 0.0
        for j in range(2*N - 1):
            w = weights[j]
            if w < 1e-15: continue
            # h[j-i]
            if 0 <= j-i < N:
                acc += w * h[j-i]
            # h[i-j]
            if 0 <= i-j < N:
                acc += w * h[i-j]
        grad_soft_max[i] = acc
        
    # Full gradient
    return 2.0 * N * (grad_soft_max * s - 2.0 * soft_max) / (s * s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _hard_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Generate diverse initializations.
    # 2. For each, run L-BFGS-B with increasing beta (annealing).
    # 3. Keep the best result.
    
    # Initializations
    initializations = []
    
    # Random Gaussian
    for _ in range(10):
        h0 = rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
        
    # Random Signs (Rademacher)
    for _ in range(10):
        h0 = rng.choice([-1.0, 1.0], size=N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
        
    # Low frequency waves
    for freq in [1, 2, 3, 5, 10]:
        t = np.linspace(0, 2*np.pi*freq, N)
        h0 = np.sin(t) + 0.5*np.cos(2*t)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)
        
    # Comb patterns
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)

    # Optimize each
    # Use a range of betas to anneal
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        h_curr = h0.copy()
        
        # Anneal beta
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                res = minimize(_soft_max_obj, h_curr,
                              jac=_soft_max_grad,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                h_new = res.x
                val_new = _hard_value(h_new)
                
                if val_new < best_val:
                    best_val = val_new
                    best_h = h_new.copy()
                    
                # Update current for next beta
                h_curr = h_new.copy()
                
            except Exception:
                pass
    
    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        for beta in [1000.0, 5000.0]:
            if time.time() > deadline - 2.0:
                break
            try:
                res = minimize(_soft_max_obj, best_h,
                              jac=_soft_max_grad,
                              method='L-BFGS-B',
                              args=(beta,),
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                val = _hard_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
            except Exception:
                pass
    
    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard_value(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
