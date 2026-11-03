import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    # Strategy: Represent the step function via low-frequency Fourier coefficients
    # to enforce global smoothness and minimize the peak-to-energy ratio of autoconvolution
    
    # Define a small set of low-frequency harmonics
    n_harmonics = 20
    # Frequencies corresponding to harmonics 1, 2, ..., n_harmonics
    # For a function on [-1/4, 1/4] with N intervals, the fundamental frequency is 1
    # Harmonics are at frequencies k for k = 1, ..., n_harmonics
    
    # We'll use a cosine series (even function) to maintain symmetry
    # f(x) = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*x)
    # x is in [-1/4, 1/4], so 2*pi*k*x ranges over [-pi*k/2, pi*k/2]
    
    # Sample points for the step function (centers of intervals)
    x = np.linspace(-0.25 + 0.25/N, 0.25 - 0.25/N, N)
    
    # Fourier basis matrix
    # B[i, 0] = 1 (DC term)
    # B[i, k] = cos(2*pi*k*x[i]) for k = 1, ..., K
    K = n_harmonics
    B = np.zeros((N, K + 1))
    B[:, 0] = 1.0
    for k in range(1, K + 1):
        B[:, k] = np.cos(2 * np.pi * k * x)
    
    # Objective function: minimize max(autoconv) / (sum(f)^2)
    # Autoconvolution of step function f on equal intervals:
    # (f * f)[m] = sum_n f[n] * f[m-n] (discrete autoconvolution)
    # Normalized: max(f*f) / (sum(f)^2)
    
    def objective(coeffs):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return np.inf
        
        # Compute f from Fourier coefficients
        f = B @ coeffs
        
        # Ensure non-negative
        f = np.maximum(f, 0)
        
        # Compute autoconvolution
        autoconv = np.convolve(f, f, mode='full')
        
        # Normalized maximum
        total = np.sum(f)
        if total < 1e-10:
            return np.inf
        return np.max(autoconv) / (total * total)
    
    # Initial guess: uniform distribution (all coefficients zero except DC)
    # For uniform f = c, autoconv has max = N*c^2, total = N*c, so ratio = N*c^2/(N*c)^2 = 1/N
    # We want to find if we can do better
    x0 = np.zeros(K + 1)
    x0[0] = 1.0  # Start with DC term
    
    # Try gradient-based optimization
    best_score = None
    best_coeffs = None
    
    # Try a few different initial points
    for init_idx in range(5):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        if init_idx == 0:
            x_init = x0.copy()
        else:
            # Small random perturbation
            x_init = x0.copy()
            x_init[1:] += 0.1 * np.random.randn(K)
        
        result = minimize(objective, x_init, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < (best_score if best_score is not None else np.inf):
            best_score = result.fun
            best_coeffs = result.x
    
    # If optimization didn't work well, fall back to a smooth window
    if best_score is None or best_score > 1.0/N:
        # Use a raised cosine (Hann window) as fallback
        f = 0.5 * (1 + np.cos(2 * np.pi * x))
        f = np.maximum(f, 0)
        return list(f)
    
    # Compute final f from best coefficients
    f = B @ best_coeffs
    f = np.maximum(f, 0)
    
    return list(f)
