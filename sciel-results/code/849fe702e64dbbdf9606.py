"""
Strategy: Model the step heights h as the absolute values of the inverse Discrete Sine Transform (DST) of a non-negative spectral envelope, ensuring h >= 0 by construction and enabling gradient-based optimization of the smooth spectral coefficients to directly flatten the autocorrelation peak g[0] relative to the tail, thereby maximizing the L2/(L1 * L_inf) ratio.

Rationale:
1. The current program uses Simulated Annealing (SA) with a smoothed objective. While effective, it is a stochastic local search method that may not efficiently navigate the complex landscape defined by the autocorrelation structure.
2. By parameterizing h via the inverse DST of a spectral envelope, we can ensure h >= 0 (by taking absolute values) and potentially exploit the frequency-domain structure of the problem.
3. The objective function involves the autocorrelation g = h * h. In the frequency domain, this corresponds to the square of the magnitude of the Fourier transform of h.
4. We use L-BFGS-B to optimize the spectral coefficients. The non-negativity constraint is handled by the absolute value, which is non-differentiable at zero. To handle this, we can use a smooth approximation like sqrt(x^2 + eps) or simply rely on the optimizer's handling of the kink, or better, optimize the squared heights directly and take the square root at the end? No, the task requires h to be non-negative step heights.
5. Let's try a different approach: Parameterize h_i = exp(s_i) where s_i are free variables. This ensures h_i > 0. Then we optimize s. The objective becomes a function of s. This is smooth everywhere.
6. We will use L-BFGS-B to maximize the exact objective (or a smoothed version if needed, but with exp parameterization, the max term in norm_inf is still non-differentiable if multiple g[k] are close to max. However, L-BFGS-B can often handle this or we can use the log-sum-exp trick again).
7. Let's combine: h_i = exp(s_i). Objective: L2(h) / (L1(g) * L_inf(g)).
   g = conv(h, h).
   This is a smooth function of s except for the max term. We will use the log-sum-exp approximation for norm_inf to make the whole objective smooth with respect to s.
"""

import numpy as np
import time
import sys

N = 50

def compute_value_smooth_s(s, beta=100.0):
    """
    Compute a smoothed version of the objective function given log-heights s.
    h = exp(s).
    Uses log-sum-exp for norm_inf to ensure differentiability.
    Uses FFT for fast convolution.
    """
    h = np.exp(s)
    
    # FFT-based convolution
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    G = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    
    # Clean up numerical noise
    G[G < 0] = 0.0
    
    m = len(G)
    
    # Construct y vector as per task:
    # y = [0, g[0], ..., g[m-1], 0]
    # y has length m+2
    y = np.zeros(m + 2)
    y[1:m+1] = G
    
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 term: sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y[:m+1] corresponds to indices 0..m
    # y[1:m+2] corresponds to indices 1..m+1
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    # norm_1 = sum(abs(g)) / (m + 1)
    norm_1 = np.sum(G) / (m + 1)
    
    # norm_inf = max(abs(g)) approximated by log-sum-exp
    # max(x) ~ (1/beta) * log(sum(exp(beta * x)))
    # To avoid overflow, subtract max
    max_g = np.max(G)
    if max_g == 0:
        return -np.inf
    norm_inf_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * (G - max_g))) + 1e-300) + max_g
    
    if norm_1 * norm_inf_smooth == 0:
        return -np.inf
        
    return l2 / (norm_1 * norm_inf_smooth)

def compute_value_exact(h):
    """
    Compute the exact objective function as defined in the task.
    Used for final validation and comparison.
    """
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return -np.inf
    h = np.clip(h, 0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return -np.inf
        
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial solution: Log-uniform or log-analytical
    # Let's start with a cosine bump in h, so s = log(h)
    x = np.linspace(0, 1, N)
    h0 = 0.5 * (1 + np.cos(2 * np.pi * x))
    h0 = np.clip(h0, 1e-6, None) # Avoid log(0)
    s0 = np.log(h0)
    
    # Optimize s using L-BFGS-B
    from scipy.optimize import minimize
    
    def neg_smooth_objective_s(s_vec):
        val = compute_value_smooth_s(s_vec, beta=100.0)
        if np.isnan(val) or val == -np.inf:
            return 1e10
        return -val
    
    best_s = s0.copy()
    best_val_smooth = compute_value_smooth_s(best_s, beta=100.0)
    best_h = np.exp(best_s)
    best_val_exact = compute_value_exact(best_h)
    
    # Run L-BFGS-B
    try:
        res = minimize(neg_smooth_objective_s, best_s, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000})
        
        candidate_s = res.x
        candidate_h = np.exp(candidate_s)
        candidate_val_smooth = compute_value_smooth_s(candidate_s, beta=100.0)
        candidate_val_exact = compute_value_exact(candidate_h)
        
        if candidate_val_exact > best_val_exact:
            best_h = candidate_h
            best_s = candidate_s
            best_val_smooth = candidate_val_smooth
            best_val_exact = candidate_val_exact
            
    except Exception as e:
        pass # If L-BFGS fails, we stick with the initial result
    
    # If we have time, try a few more random restarts in s-space
    elapsed = time.time() - start_time
    if elapsed < 100.0:
        np.random.seed(50)
        for i in range(5):
            if time.time() > deadline:
                break
            # Perturb s
            s_pert = best_s + np.random.normal(0, 0.1, N)
            try:
                res = minimize(neg_smooth_objective_s, s_pert, method='L-BFGS-B', 
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
                candidate_s = res.x
                candidate_h = np.exp(candidate_s)
                candidate_val_exact = compute_value_exact(candidate_h)
                
                if candidate_val_exact > best_val_exact:
                    best_h = candidate_h
                    best_s = candidate_s
                    best_val_exact = candidate_val_exact
            except Exception:
                pass
            
    # Final check
    final_val = compute_value_exact(best_h)
    
    # Fallback if something went wrong
    if not np.isfinite(final_val) or final_val <= 0:
        return list(np.ones(N))
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
