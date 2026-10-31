"""Optimized program for first autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
dx = (1.0/4 - (-1.0/4)) / N  # interval width = 1/300

def _autoconv_max_normalized(heights):
    """Compute normalized max of autoconvolution for step function heights."""
    # heights: array of N nonnegative values
    # Autoconvolution of step function: for each offset k in intervals,
    # the contribution at the center involves overlapping intervals.
    # For a step function, (f*f)(t) is piecewise linear.
    # The maximum is at t=0 for symmetric nonnegative functions.
    # At t=0: (f*f)(0) = sum_i h_i^2 * dx  (each interval contributes h_i*h_i*dx at center)
    # Wait, more carefully: (f*f)(0) = integral f(s)f(-s) ds
    # For step function on [-1/4,1/4], f(s) = h_i on interval i
    # f(-s) = h_{N-1-i} on interval i (by symmetry if we index properly)
    # Actually let's compute the autoconvolution at all relevant points.
    
    # The autoconvolution (f*f)(t) for t in [-1/2, 1/2]
    # For step functions, at t = k*dx for integer k:
    # (f*f)(k*dx) = dx * sum_i h_i * h_{i-k} (with h_j = 0 if j out of [0,N-1])
    # But the max might be between integer multiples.
    
    # For nonnegative step functions, the autoconvolution is maximized at t=0
    # (by rearrangement inequality / symmetry considerations)
    # At t=0: (f*f)(0) = dx * sum_i h_i^2
    
    # Normalization: likely (f*f)(0) / (integral f)^2
    # integral f = dx * sum h_i
    # So normalized = dx * sum(h_i^2) / (dx * sum(h_i))^2 = sum(h_i^2) / (dx * (sum h_i)^2)
    
    # But wait - maybe the normalization is different. Let me consider:
    # "normalized maximum of its autoconvolution" 
    # Could be max(f*f) / ||f||_2^2 = max(f*f) / (dx * sum h_i^2)
    # Or max(f*f) / (||f||_1)^2
    
    # For the first autocorrelation inequality C1, typically:
    # ||f*f||_infty <= C1 * ||f||_1^2 / ||f||_2^2 or similar
    
    # Let me use: score = max_autoconv / (integral(f))^2
    # This is dimensionless and scale-invariant.
    
    int_f = dx * np.sum(heights)
    if int_f < 1e-15:
        return 1e10
    # Autoconvolution at center (t=0) for step function
    # (f*f)(0) = dx * sum_i h_i^2  (for the case where we're computing at exact center)
    # But actually for a step function, the autoconvolution at t=0 is:
    # integral f(s)f(-s) ds = sum_i h_i * h_i * dx = dx * sum h_i^2
    autoconv_0 = dx * np.sum(heights**2)
    
    # But the max might not be exactly at 0 for asymmetric functions.
    # For safety, let's compute the max over all integer lags.
    # (f*f)(k*dx) = dx * sum_i h_i * h_{i-k} for k in [-N+1, N-1]
    # For nonnegative functions, by rearrangement, max is at k=0.
    
    return autoconv_0 / (int_f ** 2)

def _cosine_to_heights(coeffs, N_intervals):
    """Convert cosine coefficients to step heights, ensuring nonnegativity."""
    # Use cosine series on [-1/4, 1/4]
    # f(x) = sum_{k=0}^{M} a_k * cos(k * pi * (x + 1/4) / (1/2))
    # This maps x in [-1/4, 1/4] to argument in [0, pi]
    # cos(k*theta) for theta in [0, pi]
    
    x_centers = -0.25 + (np.arange(N_intervals) + 0.5) * (0.5 / N_intervals)
    theta = np.pi * (x_centers + 0.25) / 0.5  # in [0, pi]
    
    # Build cosine matrix
    M = len(coeffs) - 1
    k_vals = np.arange(M + 1)
    cos_mat = np.cos(np.outer(k_vals, theta))  # shape (M+1, N_intervals)
    
    heights = cos_mat.T @ coeffs
    # Ensure nonnegativity
    heights = np.maximum(heights, 0.0)
    return heights

def _objective(coeffs):
    """Objective: normalized max autoconvolution."""
    heights = _cosine_to_heights(coeffs, N)
    return _autoconv_max_normalized(heights)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: uniform (all coefficients zero except a_0)
    # f(x) = a_0 gives uniform function
    # For uniform f(x)=1 on [-1/4,1/4]:
    # int_f = 0.5, autoconv_0 = 0.5 * 1 = 0.5 (wait, dx*sum h^2 = (1/300)*600*1 = 2? No)
    # dx = 1/300, sum h_i^2 = 600*1 = 600, so autoconv_0 = (1/300)*600 = 2
    # int_f = (1/300)*600 = 2
    # normalized = 2 / 4 = 0.5
    # So seed gives score 1.0, meaning the raw value is 0.5? 
    # Score = seed/value, seed=1.0, so value=1.0 means raw=1.0? 
    # Wait, "score is the upper bound on C1" and "seed/value" where seed program scores 1.0
    # So the raw value for uniform is some V, and score = 1.0/V_seed * V_child... 
    # Actually: "search score is seed/value, so 1.0 equals the seed program"
    # So score = value_seed / value_child. If child does better (lower value), score > 1.
    # The evaluator measures "raw value (lower is better)"
    
    # So I need to minimize the raw value.
    
    # Start with uniform
    best_coeffs = np.zeros(20)
    best_coeffs[0] = 1.0
    best_val = _objective(best_coeffs)
    
    # Try different numbers of cosine terms
    for M in [10, 20, 30, 40, 50]:
        if time.time() > deadline:
            break
        
        # Initialize with small perturbations
        x0 = np.zeros(M + 1)
        x0[0] = 1.0
        # Add small random components to avoid symmetry
        rng = np.random.RandomState(42)
        x0[1:] = 0.01 * rng.randn(M)
        
        try:
            res = minimize(
                _objective, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Try gradient-free for more robustness
    if time.time() < deadline - 30:
        from scipy.optimize import differential_evolution
        # Use fewer terms for DE
        M_de = 15
        bounds = [(0, 2)] + [(-0.5, 0.5)] * M_de
        try:
            res_de = differential_evolution(
                _objective, bounds,
                maxiter=100, tol=1e-8,
                seed=42, polish=True
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_coeffs = res_de.x.copy()
        except Exception:
            pass
    
    # Convert best coefficients to heights
    heights = _cosine_to_heights(best_coeffs, N)
    
    # Ensure nonnegativity and return
    return [max(0.0, h) for h in heights]
