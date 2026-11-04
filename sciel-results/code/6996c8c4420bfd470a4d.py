import numpy as np
from scipy.optimize import minimize
import sys

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 40   # Increased harmonics for better approximation
    
    # Grids for optimization
    # x grid for f(x) >= 0 constraint
    n_x = 1024
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    # t grid for autoconvolution peak
    # g(t) is periodic with period L=0.5
    n_t = 2048
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    # psi_0 = sqrt(2)
    # psi_k = 2 * cos(2 * pi * k * x / L)
    
    # A: shape (n_x, M+1)
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    # B: shape (n_t, M+1)
    # g(t) = 2 * (u_0^2 * 1 + sum_{k=1} u_k^2 * cos(2 pi k t / L))
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def objective(u, smoothness=10.0):
        # Use log-sum-exp for smooth max approximation
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        # Scale for numerical stability in log-sum-exp
        max_g = np.max(g_vals)
        if max_g < 1e-10:
            return max_g, np.zeros_like(u)
        
        # Smooth max using log-sum-exp
        scaled = g_vals / smoothness
        scaled -= np.max(scaled)
        log_sum_exp = np.log(np.sum(np.exp(scaled))) + np.max(scaled)
        smooth_max = smoothness * log_sum_exp
        
        # Gradient: derivative of smooth max
        # d/dg_j [smooth_max] = exp(g_j/smoothness) / sum(exp(g_i/smoothness))
        weights = np.exp(scaled) / np.sum(np.exp(scaled))
        # d smooth_max / du_k = sum_j weights_j * d g_j / du_k
        # d g_j / du_k = 4 * u_k * B[j, k]
        grad = 4.0 * u * (weights @ B)
        
        return smooth_max, grad

    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        f_vals = A @ u
        return f_vals

    # Initial guess: Uniform function
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    # Multiple restarts to avoid local minima
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 10
    
    # Define some structured initial guesses
    initial_guesses = [u0.copy()]
    
    # Add some structured guesses: emphasize low frequencies
    for k in [1, 2, 3, 5, 8]:
        u_guess = np.zeros(M + 1)
        u_guess[0] = 0.8
        u_guess[k] = np.sqrt(1.0 - 0.64)
        initial_guesses.append(u_guess / np.linalg.norm(u_guess))
    
    # Add random starts
    np.random.seed(42)
    for _ in range(5):
        u_rand = np.random.randn(M + 1)
        u_rand[0] = abs(u_rand[0])  # Keep DC term positive
        initial_guesses.append(u_rand / np.linalg.norm(u_rand))
    
    for r, u_start in enumerate(initial_guesses):
        # Define constraints for SLSQP
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A}
        ]
        
        try:
            res = minimize(
                lambda u: objective(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective(u)[1],
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            u_opt = res.x
            # Check energy constraint
            energy = np.dot(u_opt, u_opt)
            if energy > 1e-10:
                u_opt = u_opt / np.sqrt(energy)
            
            # Check non-negativity with tolerance
            f_vals = A @ u_opt
            min_f = np.min(f_vals)
            if min_f >= -1e-4:
                # Calculate actual score (not smooth approximation)
                u_sq = u_opt**2
                g_vals = 2.0 * (B @ u_sq)
                score = np.max(g_vals)
                if score < best_score:
                    best_score = score
                    best_u = u_opt.copy()
                    
        except Exception as e:
            pass
    
    # If optimization failed or didn't improve, use best_u (which might be u0)
    # Evaluate on the final 600 step intervals
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    # Ensure non-negative
    f_centers = np.maximum(0, f_centers)
    
    return list(f_centers)

if __name__ == "__main__":
    # Quick self-check
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
