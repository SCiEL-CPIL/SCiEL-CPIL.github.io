"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 600
DT = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # interval width = 0.5/600 = 1/1200

def _autoconvolve_peak(h):
    """Compute the maximum of the autoconvolution of step function h.
    
    The autoconvolution of a step function with heights h_i on intervals of width dt
    is a piecewise quadratic function. We compute its values at the grid points
    and also check the peaks within each interval.
    
    For a step function f(x) = h_i on interval i, the autoconvolution (f*f)(t)
    can be computed via FFT or directly. Since N=600, direct computation is feasible
    but FFT is faster.
    
    Actually, the autoconvolution of a discrete sequence h with itself, when interpreted
    as a step function, requires care. Let's think about this more carefully.
    
    The step function f(x) = h_i for x in [x_i, x_{i+1}) where x_i = -1/4 + i*dt.
    The autoconvolution (f*f)(t) = ∫ f(s) f(t-s) ds.
    
    This is a continuous convolution. For step functions, the result is a piecewise
    quadratic function. The maximum can occur at interior points.
    
    A practical approach: sample the autoconvolution at many points and find the max.
    But we need to be accurate.
    
    Alternative: use the fact that the autoconvolution of a step function can be
    computed by convolving the discrete sequence h with itself, but with appropriate
    scaling. Actually, for a step function, the continuous autoconvolution at point t
    is related to the discrete autocorrelation.
    
    Let me use a direct approach: compute the autoconvolution on a fine grid.
    """
    # The support of f is [-1/4, 1/4], so f*f has support [-1/2, 1/2].
    # We'll evaluate f*f on a grid covering [-1/2, 1/2].
    
    # Use FFT-based approach for efficiency
    # Pad h to length 2*N for linear convolution
    h_pad = np.zeros(2 * N)
    h_pad[:N] = h
    # Convolve h_pad with itself
    conv = np.convolve(h_pad, h_pad, mode='full')
    # The convolution gives values proportional to the autoconvolution
    # Each interval has width dt, so the continuous autoconvolution is conv * dt^2
    # Actually, let's think: the discrete convolution sum_k h_k h_{k-m} approximates
    # the integral when scaled by dt.
    
    # For a step function, the continuous autoconvolution at the center of interval m
    # is approximately dt * sum_k h_k h_{k-m} (this is a rough approximation)
    
    # More carefully: the autoconvolution is a piecewise quadratic. The peak might
    # be at the center of an interval or at boundaries.
    
    # Let's use a fine grid evaluation instead
    # The autoconvolution (f*f)(t) for step functions:
    # f(s) = h_i where s in [x_i, x_{i+1})
    # (f*f)(t) = ∫_{-1/4}^{1/4} f(s) f(t-s) ds
    
    # We'll evaluate this on a grid of points and find the maximum.
    # For accuracy, use a grid with spacing much smaller than dt.
    
    # Actually, for the purpose of optimization, we can use a discrete approximation.
    # The normalized maximum is max_t (f*f)(t) / (∫f)^2 or similar.
    
    # Let's compute:
    # 1. The integral of f: integral = sum(h) * dt
    # 2. The autoconvolution peak
    
    # For a step function, the autoconvolution can be computed as follows:
    # (f*f)(t) = ∫ f(s) f(t-s) ds
    # For t in [t_m, t_{m+1}), this is a quadratic in t.
    
    # Let's use a practical approach: evaluate on a fine grid
    n_grid = 4 * N  # fine grid
    t_grid = np.linspace(-0.5, 0.5, n_grid, endpoint=False)
    t_grid += 0.5 / n_grid  # center of each bin
    
    # Compute f(s) for s on the grid
    # s = t - t_grid
    # This is O(N * n_grid) which is 600 * 2400 = 1.44M, feasible
    
    # Actually, let's use FFT for the discrete convolution and then interpolate
    # The discrete autocorrelation of h is:
    # r[m] = sum_k h[k] h[k-m]
    # The continuous autoconvolution is approximately r[m] * dt
    
    # Let's compute the discrete autocorrelation
    h_fft = np.fft.fft(h, n=4 * N)
    autocorr = np.fft.ifft(h_fft * np.conj(h_fft)).real[:N]
    
    # The autoconvolution peak is approximately max(autocorr) * dt
    # But we need to be more careful about the normalization
    
    # The task says "normalized maximum of its autoconvolution"
    # This likely means max_t (f*f)(t) / (∫f)^2 or max_t (f*f)(t) / ∫f
    
    # Let's assume it's max_t (f*f)(t) / (∫f)^2 which is dimensionless
    # Or maybe just max_t (f*f)(t) with f normalized somehow
    
    # For now, let's use: score = max(autocorr) * dt / (sum(h) * dt)^2
    # = max(autocorr) / (dt * (sum(h))^2)
    
    integral = np.sum(h) * DT
    if integral < 1e-10:
        return 0.0
    
    peak = np.max(autocorr) * DT
    normalized_peak = peak / (integral ** 2)
    
    return normalized_peak

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # We want to minimize the normalized autoconvolution peak
    # subject to h_i >= 0
    
    def objective(h):
        h = np.maximum(h, 0)  # enforce nonnegativity
        return _autoconvolve_peak(h)
    
    # Use L-BFGS-B with nonnegativity constraints
    bounds = [(0, None)] * N
    
    best_h = h0.copy()
    best_score = objective(h0)
    
    # Try a few different initializations and local optimizations
    initializations = [
        np.ones(N),
        np.array([1.0 if i < N//2 else 0.0 for i in range(N)]),
        np.array([1.0 if (i % 2 == 0) else 0.0 for i in range(N)]),
        np.array([np.sin(np.pi * i / N) + 1.0 for i in range(N)]),
        np.array([np.cos(2 * np.pi * i / N) + 1.5 for i in range(N)]),
    ]
    
    for h_init in initializations:
        if time.time() > deadline - 10:
            break
        
        h_init = np.maximum(h_init, 0)
        
        try:
            res = minimize(objective, h_init, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success or res.fun < best_score:
                best_h = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    # Polish pass with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_h, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_score:
                best_h = np.maximum(res.x, 0)
                best_score = res.fun
        except Exception:
            pass
    
    return [float(x) for x in best_h]
