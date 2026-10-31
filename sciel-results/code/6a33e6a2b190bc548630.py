"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # The objective is to maximize ||f||_2^2 / (||f||_1 * ||f*f||_inf)
    # For nonnegative f, we can normalize ||f||_1 = 1 (scale invariance in numerator/denominator? 
    # No: ||cf||_2^2 = c^2||f||_2^2, ||cf||_1 = c||f||_1, ||cf*cf||_inf = c^2||f*f||_inf.
    # So ratio is (c^2 ||f||_2^2) / (c ||f||_1 * c^2 ||f*f||_inf) = ||f||_2^2 / (||f||_1 ||f*f||_inf).
    # Scale invariant? Let f' = f / ||f||_1. Then ||f'||_1 = 1.
    # Ratio for f' = ||f'||_2^2 / (1 * ||f'*f'||_inf) = (||f||_2^2 / ||f||_1^2) / (||f*f||_inf / ||f||_1^2).
    # Wait, the ratio is scale invariant. So we can fix ||f||_1 = 1.
    # Then maximize ||f||_2^2 / ||f*f||_inf subject to ||f||_1 = 1, f >= 0.
    
    # Let's try a few initial shapes:
    # 1. Uniform
    # 2. Triangular (symmetric)
    # 3. Ramp (asymmetric)
    # 4. Gaussian-like
    # 5. Random
    
    def score(f):
        f = np.array(f, dtype=float)
        if np.any(f < 0):
            return -np.inf
        l1 = np.sum(f)
        if l1 == 0:
            return -np.inf
        l2_sq = np.sum(f**2)
        # Compute autoconvolution
        # f*f at lag k = sum_i f[i] f[i+k] for k>=0
        # Max over all lags
        auto = np.correlate(f, f, mode='full')
        auto_inf = np.max(auto)
        if auto_inf == 0:
            return -np.inf
        return l2_sq / (l1 * auto_inf)
    
    def neg_score(f):
        f = np.maximum(f, 0)  # enforce nonnegativity
        return -score(f)
    
    best_f = None
    best_val = -np.inf
    
    def try_candidate(f):
        nonlocal best_f, best_val
        f = np.maximum(f, 0)
        val = score(f)
        if val > best_val:
            best_val = val
            best_f = f.copy()
    
    # Initial candidates
    try_candidate(np.ones(N))  # Uniform
    try_candidate(np.linspace(0, 1, N))  # Ramp
    try_candidate(np.linspace(1, 0, N))  # Reverse ramp
    try_candidate(np.abs(np.linspace(-1, 1, N)))  # V-shape
    try_candidate(1.0 - np.abs(np.linspace(-1, 1, N)))  # Triangle
    try_candidate(np.exp(-2.0 * (np.linspace(-1, 1, N)**2)))  # Gaussian
    try_candidate(np.exp(-np.linspace(-1, 1, N)))  # Exponential decay
    try_candidate(np.exp(-2.0 * np.abs(np.linspace(-1, 1, N))))  # Laplacian
    try_candidate(np.sin(np.pi * np.linspace(0, 1, N)))  # Sine bump
    try_candidate(np.cos(np.pi/2 * np.linspace(-1, 1, N))**2)  # Cosine squared
    
    # Random initialization
    for _ in range(100):
        try_candidate(np.random.rand(N))
        if time.time() > deadline:
            break
    
    # Local optimization using Nelder-Mead (no gradient needed)
    # We'll optimize the log of the score to handle scale better, or just the score.
    # Since scale invariant, we can fix sum(f)=1.
    # Let's parameterize with simplex constraint using softmax or just project.
    
    # Better: minimize -log(score) or just -score with penalty for non-negativity.
    # Nelder-Mead doesn't handle constraints well.
    # Let's use scipy.optimize.minimize with method='L-BFGS-B' and bounds.
    
    # But score is non-smooth (max in denominator).
    # Let's try optimizing with smooth approximation or just Nelder-Mead on log(f+eps).
    
    # Actually, let's just run Nelder-Mead on the raw variables with a non-negativity penalty.
    # Or better, optimize on log-scale: f = exp(x). Then f > 0 always.
    # Objective: -score(exp(x)).
    
    def neg_score_log(x):
        f = np.exp(x)
        # Scale invariance means we can fix sum(f)=1, but let's not fix it to avoid issues.
        # Just compute score.
        return -score(f)
    
    # Start from log of best candidate
    if best_f is not None:
        x0 = np.log(best_f + 1e-10)
        try:
            res = minimize(neg_score_log, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            f_opt = np.exp(res.x)
            try_candidate(f_opt)
        except Exception:
            pass
    
    # Try a few more random restarts with optimization
    for _ in range(50):
        if time.time() > deadline:
            break
        x0 = np.random.randn(N) * 0.5
        try:
            res = minimize(neg_score_log, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7})
            f_opt = np.exp(res.x)
            try_candidate(f_opt)
        except Exception:
            pass
    
    # Final polish: coordinate descent on the best candidate
    if best_f is not None:
        f = best_f.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for delta in [0.1, 0.01, 0.001, -0.1, -0.01, -0.001, 0.5, 0.05]:
                    f_try = f.copy()
                    f_try[i] = max(0, f_try[i] + delta)
                    if score(f_try) > score(f):
                        f = f_try
                        improved = True
                        break
        try_candidate(f)
    
    # Return the best candidate as a list
    if best_f is None:
        return [1.0] * N
    
    # Normalize to ensure clean output
    best_f = np.maximum(best_f, 0)
    return list(best_f)
# EVOLVE-BLOCK-END
