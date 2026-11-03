"""
Minimize the normalized maximum of the autoconvolution of a nonnegative step function
with 600 equal intervals on [-1/4, 1/4].

Strategy:
- Use a parametric model: f(x) = sum_k a_k * exp(-b_k * (x - c_k)^2) with a_k >= 0.
  This ensures non-negativity and smoothness.
- The autoconvolution (f * f)(t) is computed via FFT for efficiency.
- The objective is max_t (f * f)(t) / (sum f)^2 (normalized).
- Use L-BFGS-B or SLSQP from scipy.optimize to minimize the peak autoconvolution.
- To handle the non-smooth max, we use a log-sum-exp approximation for the max,
  or simply sample the autoconvolution on a fine grid and maximize over it.
- Since the function is defined on 600 intervals, we can represent f as a vector of
  600 heights, but that's high-dimensional. Instead, use a low-dimensional parametric
  form.

However, the task requires returning a sequence of 600 nonnegative step heights.
So we can:
1. Optimize a low-dimensional parametric form (e.g., Gaussian mixture).
2. Sample the optimized function on the 600 intervals to get the step heights.
3. Return those heights.

Let's use a Gaussian mixture model with K components. Each component has parameters:
  a_k (amplitude, >= 0), mu_k (center), sigma_k (width).
The function is f(x) = sum_k a_k * exp(-0.5 * ((x - mu_k)/sigma_k)^2).

We'll optimize the parameters to minimize the peak of the autoconvolution.
The autoconvolution of f is (f*f)(t) = integral f(x) f(x+t) dx.
For a discrete step function with 600 intervals, we can compute the autoconvolution
via FFT.

But since we're using a parametric form, we can compute f on a fine grid, then use
FFT to get the autoconvolution, then find the max.

Let's set up the optimization.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 600
# Interval boundaries on [-1/4, 1/4]
x_min, x_max = -0.25, 0.25
dx = (x_max - x_min) / N
x_grid = np.linspace(x_min + dx/2, x_max - dx/2, N)  # centers of intervals

# For autoconvolution computation, we need to be careful with the domain.
# The autoconvolution (f*f)(t) for t in [x_min, x_max] is:
#   (f*f)(t) = integral_{max(x_min, t-x_max)}^{min(x_max, t-x_min)} f(x) f(x+t) dx
# But since f is zero outside [-1/4, 1/4], we can compute it via FFT by padding.

def compute_autoconvolution(f_vals):
    """
    Compute the autoconvolution of a step function defined by f_vals on N intervals.
    Returns the autoconvolution evaluated on a grid of 2*N-1 points (or similar).
    """
    # f_vals is an array of length N, representing the function on N intervals.
    # To compute the autoconvolution, we can use the fact that:
    #   (f*f)(t) = sum_i f_i * f_j where t = x_i - x_j (approximately)
    # But for a step function, the autoconvolution is a piecewise linear function.
    # However, for the purpose of finding the peak, we can compute it on a fine grid.
    
    # Use FFT: zero-pad f to length 2*N, then FFT, square, inverse FFT.
    f_padded = np.zeros(2 * N)
    f_padded[:N] = f_vals
    F = np.fft.fft(f_padded)
    auto = np.fft.ifft(F * np.conj(F)).real
    # The autoconvolution is defined for lags from -(N-1)*dx to (N-1)*dx.
    # The result `auto` has length 2*N. The first N values correspond to lags 0 to (N-1)*dx,
    # and the last N values correspond to lags -(N-1)*dx to -dx (due to circular convolution).
    # Actually, for linear convolution, we need to take the first 2*N-1 values appropriately.
    # Let's just take the real part and find the max over the central portion.
    # The autoconvolution at lag 0 is at index N (or 0, depending on convention).
    # Let's verify: np.fft.ifft(np.fft.fft(a) * np.fft.fft(a)) gives the circular convolution.
    # For linear convolution, we need to pad to at least len(a) + len(a) - 1.
    # We padded to 2*N, which is enough for linear convolution of two length-N signals.
    # The result of ifft has length 2*N. The linear convolution result is in the first 2*N-1 elements,
    # but wrapped around. Actually, for real signals, the autoconvolution is symmetric.
    # Let's just take the max over the entire array, but note that the values at the ends
    # are small (since the function is zero outside).
    
    # Actually, the standard way: 
    #   conv = np.convolve(f_vals, f_vals, mode='full')
    # This gives the linear convolution, which is the autoconvolution evaluated at 
    # lags from -(N-1)*dx to (N-1)*dx.
    # Let's use np.convolve for simplicity and correctness.
    conv = np.convolve(f_vals, f_vals, mode='full')
    return conv

def objective(params):
    """
    params: array of length 3*K, where K is the number of Gaussian components.
            For each component: [a_k, mu_k, sigma_k]
    Returns: the normalized peak autoconvolution.
    """
    K = len(params) // 3
    a = params[0::3]
    mu = params[1::3]
    sigma = params[2::3]
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Evaluate f on the N intervals
    f_vals = np.zeros(N)
    for k in range(K):
        f_vals += a[k] * np.exp(-0.5 * ((x_grid - mu[k]) / sigma[k])**2)
    
    # Ensure non-negativity (should be guaranteed by a_k >= 0, but clip just in case)
    f_vals = np.maximum(f_vals, 0.0)
    
    # Compute autoconvolution
    conv = compute_autoconvolution(f_vals)
    
    # The peak of the autoconvolution
    peak = np.max(conv)
    
    # Normalize by (sum f_vals * dx)^2
    integral = np.sum(f_vals) * dx
    if integral < 1e-12:
        return 1e12
    normalized_peak = peak / (integral * integral)
    
    return normalized_peak

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds max
    
    # Try a few different initializations and pick the best.
    best_result = None
    best_score = np.inf
    
    # Number of Gaussian components to try
    K_list = [1, 2, 3]
    
    for K in K_list:
        if time.time() > deadline - 10:
            break
        
        # Initial parameters
        # a_k: positive, mu_k: in [-0.25, 0.25], sigma_k: positive
        np.random.seed(42)
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            a_init = np.random.uniform(0.5, 2.0, K)
            mu_init = np.random.uniform(-0.2, 0.2, K)
            sigma_init = np.random.uniform(0.05, 0.2, K)
            
            params_init = np.concatenate([a_init, mu_init, sigma_init])
            
            # Bounds
            bounds = []
            for k in range(K):
                bounds.append((0.0, 10.0))  # a_k
                bounds.append((-0.25, 0.25))  # mu_k
                bounds.append((0.01, 0.5))  # sigma_k
            
            # Optimize
            try:
                result = minimize(
                    objective,
                    params_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if result.success or result.fun < best_score:
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_result = result.x
            except Exception:
                pass
        
        # Also try a uniform initialization
        a_init = np.ones(K)
        mu_init = np.zeros(K)
        sigma_init = np.full(K, 0.15)
        params_init = np.concatenate([a_init, mu_init, sigma_init])
        
        bounds = []
        for k in range(K):
            bounds.append((0.0, 10.0))
            bounds.append((-0.25, 0.25))
            bounds.append((0.01, 0.5))
        
        try:
            result = minimize(
                objective,
                params_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                score = result.fun
                if score < best_score:
                    best_score = score
                    best_result = result.x
        except Exception:
            pass
    
    # If no optimization succeeded, fall back to a simple heuristic
    if best_result is None:
        # Return a constant function
        return [1.0] * N
    
    # Convert the optimized parameters to step heights
    K = len(best_result) // 3
    a = best_result[0::3]
    mu = best_result[1::3]
    sigma = best_result[2::3]
    sigma = np.maximum(sigma, 1e-6)
    
    f_vals = np.zeros(N)
    for k in range(K):
        f_vals += a[k] * np.exp(-0.5 * ((x_grid - mu[k]) / sigma[k])**2)
    
    f_vals = np.maximum(f_vals, 0.0)
    
    # Normalize so that the integral is 1 (optional, but the score is normalized)
    integral = np.sum(f_vals) * dx
    if integral > 0:
        f_vals = f_vals / integral
    
    return list(f_vals)
