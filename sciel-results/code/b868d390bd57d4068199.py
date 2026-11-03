import numpy as np
from scipy.optimize import minimize
import sys

N = 600

def construct():
    L = 0.5
    n_grid = 4096
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    dx = L / n_grid
    
    M = 15
    
    # Precompute basis matrices for speed
    # f(x) = sum_{k=0}^{M} u_k * phi_k(x)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2*pi*k*x/L) for k>=1
    
    basis = np.zeros((n_grid, M+1))
    basis[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        basis[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # We want to minimize max(g(t)) where g(t) is the autoconvolution.
    # g(t) = (f*f)(t)
    # If we assume the function is band-limited to M harmonics and periodic,
    # g(t) = sum_{k=-M}^{M} a_k^2 e^{i 2pi k t / L}
    # With real even f, a_k = a_{-k}.
    # a_0 = u_0 / sqrt(L)
    # a_k = u_k / sqrt(2L) for k>0
    # g(t) = a_0^2 + 2 * sum_{k=1}^{M} a_k^2 cos(2*pi*k*t/L)
    #      = u_0^2/L + sum_{k=1}^{M} (u_k^2/L) cos(2*pi*k*t/L)
    
    # However, this periodic assumption ignores the truncation.
    # The task specifies a step function on [-1/4, 1/4].
    # The true autoconvolution is the convolution of the truncated function.
    # Let's compute the true autoconvolution using FFT for accuracy.
    
    # We will optimize u such that:
    # 1. ||u||_2 = 1 (normalizes energy to 1)
    # 2. f(x) >= 0 for all x in [-L/2, L/2]
    # 3. min over t of max(f*f)(t)
    
    # To handle the non-negativity constraint and the max operation,
    # we will use a penalty method with gradient descent on the sphere.
    
    def compute_f(u):
        return basis.dot(u)
    
    def compute_convolution_max(u):
        # Compute f on a finer grid for convolution?
        # Actually, f is defined on x.
        # We can compute the convolution via FFT.
        # Zero-pad f to length n_fft = 2*n_grid to avoid circular convolution issues.
        f_vals = compute_f(u)
        n_fft = 2 * n_grid
        f_fft = np.zeros(n_fft)
        f_fft[:n_grid] = f_vals
        
        F = np.fft.rfft(f_fft)
        C = np.fft.irfft(F * F, n=n_fft)
        
        # The convolution integral is approximated by sum C * dx
        # But we need the continuous max.
        # The discrete max is a good approximation if n_grid is large enough.
        max_conv = np.max(C) * dx
        return max_conv
    
    def objective(u):
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-10:
            return 1e10
        u_norm = u / norm_u
        f_vals = compute_f(u_norm)
        
        # Penalty for negative values
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        max_conv = compute_convolution_max(u_norm)
        
        return max_conv + 100.0 * penalty
    
    def gradient(u):
        # Numerical gradient is safer given the complex objective
        eps = 1e-5
        grad = np.zeros_like(u)
        for i in range(len(u)):
            u_plus = u.copy()
            u_plus[i] += eps
            u_minus = u.copy()
            u_minus[i] -= eps
            
            # Normalize for objective?
            # The objective function internally normalizes.
            # But for gradient of the normalized objective, we need to be careful.
            # Let's just use the raw gradient of the unnormalized function and project?
            # No, let's stick to the penalty method on the sphere.
            
            # Actually, let's use a simpler approach:
            # Minimize F(u) = max_conv(u/||u||) + penalty(u/||u||)
            # This is differentiable almost everywhere.
            
            # For speed, let's approximate the gradient of the objective.
            # But numerical gradient with FFT is expensive (2*M+2 FFTs per iteration).
            # M=15, so ~32 FFTs. n_grid=4096.
            # 32 * 4096 log 4096 ~ 32 * 50k ~ 1.6M ops.
            # 100 iterations * 1.6M = 160M ops. Should be fine in 120s.
            
            grad[i] = (objective(u_plus) - objective(u_minus)) / (2*eps)
        
        return grad
    
    # Initial guess
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_u = u0.copy()
    best_score = objective(u0)
    
    np.random.seed(42)
    
    # Try a few random restarts
    for restart in range(5):
        if restart == 0:
            u = u0.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
        
        # Gradient descent on the sphere
        lr = 0.01
        for i in range(200):
            # Normalize u
            u_norm = u / np.linalg.norm(u)
            
            # Compute gradient
            grad = gradient(u_norm)
            
            # Project gradient to tangent space of sphere
            # grad_tangent = grad - (grad . u) * u
            grad_tangent = grad - np.dot(grad, u_norm) * u_norm
            
            # Update
            u = u_norm - lr * grad_tangent
            
            # Check convergence
            if i % 10 == 0:
                score = objective(u)
                if score < best_score:
                    best_score = score
                    best_u = u.copy()
        
        # Final check
        score = objective(u)
        if score < best_score:
            best_score = score
            best_u = u.copy()
    
    # Ensure best_u is normalized
    best_u /= np.linalg.norm(best_u)
    
    # Compute heights on the 600 intervals
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    # f(x) = sum u_k phi_k(x)
    heights = np.zeros(N)
    heights += best_u[0] / np.sqrt(L)
    for k in range(1, M+1):
        heights += best_u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*centers/L)
    
    # Ensure non-negativity
    heights = np.maximum(0, heights)
    
    return list(heights)
