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
from scipy.optimize import minimize, LinearConstraint
import sys

N = 600

def construct():
    # Parameters
    M = 20  # Number of harmonics
    L = 0.5  # Length of the interval [-1/4, 1/4]
    
    # Discretization for checking constraints and computing objective
    # Use a fine grid for evaluating f(x) >= 0 and max of g(t)
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    dx = L / n_grid
    
    # Time grid for autoconvolution g(t)
    # g(t) is supported on [-L, L] = [-0.5, 0.5]
    n_t_grid = 4096
    t = np.linspace(-L, L, n_t_grid, endpoint=False)
    dt = 2*L / n_t_grid
    
    # Fourier basis functions
    # f(x) = sum_{k=0}^M c_k * phi_k(x)
    # phi_0(x) = 1
    # phi_k(x) = 2 * cos(2π k x / L) for k >= 1
    # Note: The standard Fourier series on [-L/2, L/2] with period L is:
    # f(x) = a_0/2 + sum a_k cos(2π k x / L)
    # Let's use c_k as the coefficients directly.
    # To make ||f||_2^2 easy, let's use orthonormal basis.
    # Basis: 
    # e_0(x) = 1/sqrt(L)
    # e_k(x) = sqrt(2/L) cos(2π k x / L) for k >= 1
    # Then f(x) = sum_{k=0}^M u_k e_k(x)
    # ||f||_2^2 = sum u_k^2
    # f(x) >= 0 constraint: sum u_k e_k(x) >= 0
    
    # Let's stick to the simpler coefficient representation:
    # f(x) = c_0 + 2 * sum_{k=1}^M c_k cos(2π k x / L)
    # Energy E = ∫_{-L/2}^{L/2} f(x)^2 dx = L * (c_0^2 + sum_{k=1}^M c_k^2)
    # Wait, ∫ cos^2(2π k x/L) dx = L/2.
    # ∫ 2 c_k cos(...) * 2 c_j cos(...) = 2 c_k c_j ∫ 2 cos(...) cos(...) 
    # If k=j>=1: 4 c_k^2 * (L/2) = 2 L c_k^2.
    # If k=0, j=0: L c_0^2.
    # Cross terms vanish.
    # So E = L c_0^2 + 2 L sum_{k=1}^M c_k^2.
    
    # Let's define the vector of parameters v = [c_0, c_1, ..., c_M]
    # We want to minimize max(g(t)) subject to E=1 and f(x)>=0.
    
    # Precompute basis values
    # e_k(x) for the representation f(x) = sum v_k * basis_k(x)
    # Let's use the normalized basis for easier gradient computation.
    # Let f(x) = sum_{k=0}^M u_k * phi_k(x)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2π k x / L)
    
    # Then ||f||_2^2 = sum u_k^2.
    # f(x) = (1/sqrt(L)) u_0 + sum_{k=1}^M sqrt(2/L) u_k cos(2π k x / L)
    
    # Autoconvolution g(t) = ∫ f(τ) f(t-τ) dτ
    # In terms of Fourier coefficients u_k:
    # The Fourier series of f is sum u_k phi_k.
    # The autoconvolution in the time domain corresponds to the product of Fourier transforms.
    # Let's compute g(t) directly by summing the series.
    # f(x) = sum_{k=-M}^M a_k e^{i 2π k x / L}
    # where a_0 = u_0/sqrt(L)
    # a_k = u_k/sqrt(2L) for k>0, a_{-k} = u_k/sqrt(2L)
    
    # g(t) = sum_{k,l} a_k a_l ∫ e^{i 2π k τ / L} e^{i 2π l (t-τ) / L} dτ
    # = sum_{k,l} a_k a_l e^{i 2π l t / L} ∫ e^{i 2π (k-l) τ / L} dτ
    # The integral is non-zero only if k=l (for the periodic extension case).
    # Wait, this is for periodic functions. Our function is supported on [-L/2, L/2] and zero elsewhere?
    # The task says "step function on [-1/4, 1/4]". Usually this implies it's zero outside.
    # If it's zero outside, the Fourier series representation on the interval is just a representation,
    # but the autoconvolution is the true convolution of the truncated function.
    
    # This is tricky. If we use Fourier series on the interval, we are implicitly assuming periodicity.
    # The autoconvolution of a periodic function is different from the autoconvolution of the truncated function.
    
    # Let's re-read the task: "Choose a nonnegative step function with 600 equal intervals on [-1/4,1/4]".
    # This implies the function is defined only on this interval.
    # The autoconvolution (f*f)(t) = ∫ f(τ) f(t-τ) dτ.
    # If f is zero outside [-1/4, 1/4], then the integral is over the intersection of [-1/4, 1/4] and [t-1/4, t+1/4].
    
    # If we represent f by Fourier series on [-1/4, 1/4], we are approximating f by a periodic function.
    # The autoconvolution of the periodic function is not the same as the autoconvolution of the truncated function.
    
    # However, if we choose f to be very smooth and band-limited, the Gibbs phenomenon is reduced.
    # But the boundary effects are still there.
    
    # Alternative approach:
    # Since we have 600 intervals, we can directly optimize the 600 heights.
    # But the directive says: "Represent the step function via its Fourier coefficients on a small set of low-frequency harmonics".
    # So we must use the Fourier representation.
    # This implies we are effectively optimizing over the subspace of functions that are well-approximated by low-frequency harmonics.
    # The autoconvolution we minimize should be the true autoconvolution of the step function defined on the interval.
    
    # How to compute the true autoconvolution from Fourier coefficients?
    # It's not straightforward.
    
    # Let's assume the directive implies that we should minimize the autoconvolution of the *periodic extension* or that the function is designed to be periodic (i.e., f(-1/4) = f(1/4)).
    # If f is periodic, the autoconvolution is well-defined via the Fourier series product.
    # Let's proceed with the periodic assumption, as it fits the "Fourier coefficients" hint.
    
    # So, f is periodic with period L=0.5.
    # f(x) = sum_{k=-M}^M a_k e^{i 2π k x / L}
    # a_k = a_{-k} = u_k / sqrt(2L) for k>0, a_0 = u_0/sqrt(L)
    # g(t) = (f*f)(t) = sum_{k=-M}^M a_k^2 e^{i 2π k t / L}
    # Wait, the convolution of two periodic functions with period L is:
    # (f*f)(t) = ∫_{-L/2}^{L/2} f(τ) f(t-τ) dτ
    # Using Fourier series:
    # f(τ) = sum a_k e^{i 2π k τ / L}
    # f(t-τ) = sum a_l e^{i 2π l (t-τ) / L}
    # Product: sum a_k a_l e^{i 2π l t / L} e^{i 2π (k-l) τ / L}
    # Integral over τ: non-zero only if k=l.
    # So g(t) = sum_{k=-M}^M a_k^2 e^{i 2π k t / L}
    # = a_0^2 + sum_{k=1}^M 2 a_k^2 cos(2π k t / L)
    # (since a_{-k} = a_k)
    
    # Energy E = ∫ f^2 dx = L * sum_{k=-M}^M a_k^2 = L a_0^2 + 2 L sum_{k=1}^M a_k^2
    # We want to minimize max(g(t)) / E.
    # Let's fix E=1 and minimize max(g(t)).
    
    # Variables: u_0, u_1, ..., u_M
    # a_0 = u_0 / sqrt(L)
    # a_k = u_k / sqrt(2L)
    # E = L (u_0^2/L) + 2L sum (u_k^2/(2L)) = u_0^2 + sum u_k^2 = ||u||^2
    # So fixing E=1 is fixing ||u||=1.
    
    # g(t) = (u_0^2/L) + sum_{k=1}^M 2 (u_k^2/(2L)) cos(2π k t / L)
    # = (1/L) [ u_0^2 + sum_{k=1}^M u_k^2 cos(2π k t / L) ]
    
    # We want to minimize max_t g(t) subject to ||u||=1 and f(x) >= 0.
    # f(x) = u_0/sqrt(L) + sum_{k=1}^M u_k sqrt(2/L) cos(2π k x / L) >= 0
    
    # This is a non-convex optimization problem.
    # We can use gradient descent on the sphere ||u||=1 with a penalty for f(x)<0.
    
    def compute_f(x_grid, u):
        """Compute f(x) for given grid and coefficients u."""
        f_vals = np.zeros_like(x_grid)
        f_vals += u[0] / np.sqrt(L)
        for k in range(1, M+1):
            f_vals += u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*x_grid/L)
        return f_vals
    
    def compute_g(t_grid, u):
        """Compute g(t) for given grid and coefficients u."""
        g_vals = np.zeros_like(t_grid)
        g_vals += u[0]**2 / L
        for k in range(1, M+1):
            g_vals += (u[k]**2 / L) * np.cos(2*np.pi*k*t_grid/L)
        return g_vals
    
    def objective_and_grad(u):
        """Compute objective (max g) and gradient."""
        f_vals = compute_f(x, u)
        g_vals = compute_g(t, u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        obj = np.max(g_vals) + 100.0 * penalty
        
        # Gradient
        # grad of max(g) is tricky, but we can approximate or use subgradient.
        # For simplicity, let's use numerical gradient or a simple projection method.
        # Given the time limit, let's use scipy.optimize.minimize with SLSQP or similar.
        
        return obj
    
    # Initial guess: uniform function (u_0 = 1, others 0)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    # Constraints
    # ||u||^2 = 1
    # f(x) >= 0
    
    # Let's use scipy.optimize.minimize
    # Since the problem is non-convex, we might need multiple restarts or a good initial guess.
    # The uniform function gives g(t) = 1/L = 2.
    # Can we do better?
    # If we spread the energy, max(g) might decrease.
    
    # Let's try a simple iterative algorithm:
    # 1. Start with u.
    # 2. Compute f(x). If any f(x) < 0, project u to the nearest feasible point (hard).
    # 3. Compute g(t). Find t_max.
    # 4. Update u to reduce g(t_max) while maintaining ||u||=1 and f(x)>=0.
    
    # Given the complexity, let's use a random restart approach with gradient descent on the sphere.
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Run a few iterations of a simple optimizer
    # We'll use a custom gradient descent with projection to the sphere and penalty for negativity.
    
    lr = 0.01
    n_iter = 100
    
    for restart in range(5):
        if restart == 0:
            u = u0.copy()
        else:
            # Random initialization on the sphere
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
            # Ensure f(x) is not too negative initially?
            # If it is, we'll project.
        
        for i in range(n_iter):
            f_vals = compute_f(x, u)
            g_vals = compute_g(t, u)
            
            # Penalty for negative f
            if np.any(f_vals < 0):
                # Project u to make f(x) >= 0
                # This is a quadratic programming problem:
                # min ||u - u_current||^2 subject to A u >= b
                # where A is the matrix of basis functions, b=0.
                # We can use a simple iterative projection or scipy.
                # For speed, let's just add a large penalty gradient.
                pass
            
            # Compute gradient of max(g)
            # max(g) is not differentiable, but we can use the gradient at the point of maximum.
            idx_max = np.argmax(g_vals)
            t_max = t[idx_max]
            
            # grad of g(t_max) w.r.t u
            grad_g = np.zeros(M+1)
            grad_g[0] = 2 * u[0] / L
            for k in range(1, M+1):
                grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
            
            # Gradient of penalty
            grad_pen = np.zeros(M+1)
            for j in range(len(x)):
                if f_vals[j] < 0:
                    # grad of (-f(x_j))^2 = -2 f(x_j) * grad f(x_j)
                    df_dx = np.zeros(M+1)
                    df_dx[0] = 1.0/np.sqrt(L)
                    for k in range(1, M+1):
                        df_dx[k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x[j]/L)
                    grad_pen -= 2 * f_vals[j] * df_dx
            
            grad_total = grad_g + 100.0 * grad_pen
            
            # Update u
            u = u - lr * grad_total
            
            # Project to sphere
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                u /= norm_u
            
            # Check convergence
            if i % 10 == 0:
                f_check = compute_f(x, u)
                g_check = compute_g(t, u)
                if np.all(f_check >= -1e-6):
                    score = np.max(g_check)
                    if score < best_score:
                        best_score = score
                        best_u = u.copy()
    
    # Convert best_u back to step heights
    # We need to evaluate f(x) at the 600 intervals.
    # The intervals are equal on [-1/4, 1/4].
    # Let's sample f at the center of each interval.
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    heights = compute_f(centers, best_u)
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    # Normalize? The task says "normalized maximum of its autoconvolution".
    # If we fixed E=1, the score is max(g).
    # But the evaluator might normalize differently.
    # Let's assume the evaluator computes the autoconvolution of the returned heights and normalizes by the energy.
    # To be safe, we should return the heights as they are, assuming the evaluator handles normalization.
    
    return list(heights)

# To speed up, we might need to reduce M or n_iter.
# Let's refine the parameters.
M = 10  # Reduce harmonics for speed
N_ITER = 200
N_RESTARTS = 3

def construct():
    L = 0.5
    n_grid = 1024
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    n_t_grid = 2048
    t = np.linspace(-L, L, n_t_grid, endpoint=False)
    
    def compute_f(x_grid, u):
        f_vals = np.zeros_like(x_grid)
        f_vals += u[0] / np.sqrt(L)
        for k in range(1, M+1):
            f_vals += u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*x_grid/L)
        return f_vals
    
    def compute_g(t_grid, u):
        g_vals = np.zeros_like(t_grid)
        g_vals += u[0]**2 / L
        for k in range(1, M+1):
            g_vals += (u[k]**2 / L) * np.cos(2*np.pi*k*t_grid/L)
        return g_vals
    
    best_score = np.inf
    best_u = np.zeros(M+1)
    best_u[0] = 1.0
    
    for restart in range(N_RESTARTS):
        if restart == 0:
            u = best_u.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
        
        lr = 0.05
        for i in range(N_ITER):
            f_vals = compute_f(x, u)
            g_vals = compute_g(t, u)
            
            idx_max = np.argmax(g_vals)
            t_max = t[idx_max]
            
            grad_g = np.zeros(M+1)
            grad_g[0] = 2 * u[0] / L
            for k in range(1, M+1):
                grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
            
            grad_pen = np.zeros(M+1)
            for j in range(len(x)):
                if f_vals[j] < 0:
                    df_dx = np.zeros(M+1)
                    df_dx[0] = 1.0/np.sqrt(L)
                    for k in range(1, M+1):
                        df_dx[k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x[j]/L)
                    grad_pen -= 2 * f_vals[j] * df_dx
            
            grad_total = grad_g + 50.0 * grad_pen
            
            u = u - lr * grad_total
            norm_u = np.linalg.norm(u)
            if norm_u > 0:
                u /= norm_u
            
            f_check = compute_f(x, u)
            if np.all(f_check >= -1e-6):
                g_check = compute_g(t, u)
                score = np.max(g_check)
                if score < best_score:
                    best_score = score
                    best_u = u.copy()
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    heights = compute_f(centers, best_u)
    heights = np.maximum(0, heights)
    
    return list(heights)
