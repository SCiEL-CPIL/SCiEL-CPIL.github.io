import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns: (1/beta) * log(sum(exp(beta * conv)))
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    params: flattened array of h values (non-negative enforced via softplus or projection)
    """
    h = np.maximum(params, 0.0)  # Enforce non-negativity
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Objective function with numerical gradient for scipy.optimize.
    """
    return objective(params, beta)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Initialize with uniform distribution
    h_uniform = np.ones(N) / N
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # Try multiple initialization strategies
    initializations = [
        h_uniform,
        np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/4)**2)),  # Gaussian
        np.ones(N) * np.linspace(0.5, 1.5, N),  # Linear ramp
        np.ones(N) * np.sin(np.pi * np.arange(N) / N) + 0.5,  # Sine wave
        np.concatenate([np.ones(N//2) * 1.2, np.ones(N - N//2) * 0.8]),  # Step
    ]
    
    # Normalize initializations to have same sum as uniform
    for i, h_init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        h_init = h_init / np.sum(h_init) * np.sum(h_uniform)
        
        # Run gradient-based optimization
        try:
            result = minimize(
                objective,
                h_init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = np.maximum(result.x, 0.0)
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception as e:
            pass
    
    # Final local polish with coordinate descent
    if time.time() < deadline - 10:
        np.random.seed(123)
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(0, N)
            target = np.random.randint(0, N)
            if idx == target:
                continue
            
            # Try moving a small fraction of mass
            for frac in [0.01, 0.05, 0.1, 0.2]:
                delta = best_h[idx] * frac
                if delta < 0 or best_h[idx] - delta < 0:
                    continue
                h_trial = best_h.copy()
                h_trial[idx] -= delta
                h_trial[target] += delta
                score_trial = compute_score(h_trial)
                
                if score_trial < best_score:
                    best_score = score_trial
                    best_h = h_trial
                    break
    
    # Ensure non-negativity
    best_h = np.maximum(best_h, 0.0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
