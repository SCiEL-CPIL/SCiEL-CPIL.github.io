"""
Optimized program for the First Autocorrelation Inequality task.

Approach:
1. Represent the step function by its first M Fourier cosine coefficients (symmetric, even function).
2. The autoconvolution of a function f is related to the square of its Fourier transform.
3. For a periodic function on [-1/4, 1/4] with period 1/2, the autoconvolution's peak can be bounded
   by the ratio of the L2 norm of the Fourier transform to the DC component squared, but more directly,
   we minimize the maximum of the autoconvolution relative to the energy.
   
   Actually, the autoconvolution (f*f)(t) = ∫ f(τ)f(t-τ) dτ. For a function supported on [-1/4, 1/4],
   the autoconvolution is supported on [-1/2, 1/2].
   
   The task asks to minimize the "normalized maximum of its autoconvolution". 
   Let's assume the normalization is by the L2 norm of f, or by the area.
   Typically, C1 is related to the constant in the inequality ||f*f||_∞ ≤ C1 ||f||_2^2.
   So we want to minimize ||f*f||_∞ / ||f||_2^2.
   
   Using Parseval's theorem: ||f*f||_2^2 = (2π)^2 ||F||_2^2 ... wait, that's L2 norm of the convolution.
   We need the L∞ norm.
   
   However, if we restrict to band-limited functions (low frequency harmonics), the autoconvolution will also be band-limited.
   We can compute the autoconvolution in the time domain by summing the Fourier series of the product of coefficients.
   
   Let f(x) = sum_{k=-M}^{M} c_k e^{i 2π k x / L} where L = 1/2.
   Since f is real and even, c_{-k} = c_k and c_k is real. Let c_k for k>=0.
   f(x) = c_0 + 2 * sum_{k=1}^{M} c_k cos(2π k x / L).
   
   The autoconvolution g(t) = (f*f)(t).
   In the Fourier domain, G(ω) = F(ω)^2.
   Since f is band-limited to M harmonics, g is band-limited to 2M harmonics.
   We can compute g(t) at many points in time to estimate its maximum.
   
   Objective: minimize max_t g(t) / (∫ f^2 dx).
   By scaling invariance, we can fix ∫ f^2 dx = 1 and minimize max_t g(t).
   
   We will optimize the coefficients c_0, c_1, ..., c_M subject to:
   1. f(x) >= 0 for all x in [-L/2, L/2] (nonnegative step function constraint).
   2. ||f||_2^2 = 1.
   
   We use gradient descent with projection or penalty methods.
"""

import numpy as np
from scipy.optimize import minimize
import sys

N = 600

def construct():
    # Parameters
    M = 15  # Number of harmonics
    L = 0.5  # Length of the interval [-1/4, 1/4]
    
    # Discretization for checking constraints and computing objective
    # Use a fine grid for evaluating f(x) >= 0 and max of g(t)
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    
    # Time grid for autoconvolution g(t)
    # g(t) is supported on [-L, L] = [-0.5, 0.5]
    n_t_grid = 4096
    t = np.linspace(-L, L, n_t_grid, endpoint=False)
    
    # Precompute basis function values
    # f(x) = u_0 / sqrt(L) + sum_{k=1}^M u_k * sqrt(2/L) * cos(2π k x / L)
    # Let's precompute the basis matrix B where B[i, k] = phi_k(x_i)
    B = np.zeros((n_grid, M+1))
    B[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        B[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # For g(t):
    # g(t) = (1/L) [ u_0^2 + sum_{k=1}^M u_k^2 cos(2π k t / L) ]
    # Precompute C where C[i, k] = cos(2π k t_i / L) for k >= 1
    C = np.zeros((n_t_grid, M))
    for k in range(1, M+1):
        C[:, k-1] = np.cos(2*np.pi*k*t/L)
    
    def compute_f(u):
        """Compute f(x) for given coefficients u."""
        return B @ u
    
    def compute_g(u):
        """Compute g(t) for given coefficients u."""
        g_vals = np.full(n_t_grid, u[0]**2 / L)
        for k in range(1, M+1):
            g_vals += (u[k]**2 / L) * C[:, k-1]
        return g_vals
    
    def objective(u):
        """Compute objective (max g + penalty for negative f)."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = 50.0 * np.sum(neg_f**2)
        
        return np.max(g_vals) + penalty
    
    def gradient(u):
        """Compute gradient of objective."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Gradient of max(g)
        idx_max = np.argmax(g_vals)
        t_max = t[idx_max]
        
        grad = np.zeros(M+1)
        grad[0] = 2.0 * u[0] / L
        for k in range(1, M+1):
            grad[k] = (2.0 * u[k] / L) * np.cos(2*np.pi*k*t_max/L)
        
        # Gradient of penalty
        neg_mask = f_vals < 0
        if np.any(neg_mask):
            f_neg = np.where(neg_mask, f_vals, 0)
            grad_penalty = -2.0 * (B @ f_neg)
            grad += 50.0 * grad_penalty
        
        return grad
    
    # Initial guess: uniform function (u_0 = 1, others 0)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Try multiple restarts with different initial conditions
    n_restarts = 8
    
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            np.random.seed(restart * 42)
            # Random initialization with emphasis on low frequencies
            u = np.random.randn(M+1)
            u[0] = abs(u[0]) + 1.0  # Ensure DC component is positive
            # Decay higher frequencies
            for k in range(1, M+1):
                u[k] *= 0.5 ** (k / M)
            u /= np.linalg.norm(u)
        
        # Use L-BFGS-B for optimization
        result = minimize(objective, u, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_score:
            f_check = compute_f(result.x)
            if np.all(f_check >= -1e-6):
                g_check = compute_g(result.x)
                score = np.max(g_check)
                if score < best_score:
                    best_score = score
                    best_u = result.x.copy()
    
    # Convert best_u back to step heights
    # We need to evaluate f(x) at the 600 intervals.
    # The intervals are equal on [-1/4, 1/4].
    # Let's sample f at the center of each interval.
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Compute f at centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] / np.sqrt(L)
    for k in range(1, M+1):
        f_centers += best_u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*centers/L)
    
    # Ensure nonnegativity
    heights = np.maximum(0, f_centers)
    
    return list(heights)
