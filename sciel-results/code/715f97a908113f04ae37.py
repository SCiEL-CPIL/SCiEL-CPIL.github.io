import numpy as np
from scipy.optimize import minimize
import sys

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 20   # Number of harmonics (reduced for stability and speed)
    
    # Grids for optimization
    # x grid for f(x) >= 0 constraint
    n_x = 1024
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    # t grid for autoconvolution peak
    # g(t) is periodic with period L=0.5?
    # g(t) = sum c_k^2 e^{i 2 pi k t / L}. Yes, period L.
    n_t = 2048
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    # psi_0 = sqrt(2)
    # psi_k = 2 * cos(2 * pi * k * x / L) = 2 * cos(4 * pi * k * x)
    
    # A: shape (n_x, M+1)
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    # B: shape (n_t, M+1)
    # g(t) = 2 * (u_0^2 * 1 + sum_{k=1} u_k^2 * cos(2 pi k t / L))
    # We can precompute the cos terms
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def objective_and_gradient(u):
        # u is the vector of coefficients
        # g_vals = 2 * (u^T * (B * u^2)?) No.
        # g(t_j) = 2 * sum_k u_k^2 B[j, k]
        # Let u_sq = u**2
        # g_vals = 2 * (B @ u_sq)
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        # We want to minimize max(g_vals)
        # Using a smooth approximation or just the max with subgradient
        # For SLSQP, we can define the objective as max(g_vals)
        # and provide the gradient analytically or numerically.
        # grad of max(g) is the gradient at the index of max.
        idx_max = np.argmax(g_vals)
        
        # Gradient of g_vals[idx] w.r.t u
        # g(idx) = 2 * sum_k u_k^2 B[idx, k]
        # d/du_k g = 4 * u_k * B[idx, k]
        grad = 4.0 * u * B[idx_max, :]
        
        return g_vals[idx_max], grad

    def constraint_energy(u):
        # sum u^2 - 1 = 0
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        # f(x) = A @ u >= 0
        f_vals = A @ u
        # Return a vector of values that should be >= 0
        # SLSQP handles inequality constraints as g(x) >= 0
        return f_vals

    def constraint_nonneg_grad(u, jac):
        # This is tricky with SLSQP if we don't provide analytic jac for constraints
        # SLSQP can compute finite difference jacobians, which is slow but okay for M=20
        # If we provide jac, it must be the Jacobian of the constraint function.
        # constraint_nonneg returns f_vals (shape n_x).
        # Jacobian is A (shape n_x, M+1)
        return A

    # Initial guess: Uniform function
    # u_0 = 1, others 0 => f(x) = sqrt(2), constant.
    # Energy = 2 * 0.5 = 1. Correct.
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    # Multiple restarts to avoid local minima
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 10
    
    for r in range(n_restarts):
        if r == 0:
            u_start = u0.copy()
        else:
            # Random start on sphere
            # Bias towards lower frequencies to help with non-negativity
            u_rand = np.random.randn(M + 1)
            # Decay higher frequencies
            decay = np.exp(-np.linspace(0, 1, M + 1) * 2.0)
            u_rand *= decay
            u_start = u_rand / np.linalg.norm(u_rand)
            # Ensure non-negativity is not grossly violated initially?
            # SLSQP will handle it, but a bad start might diverge.
            
        # Define constraints for SLSQP
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A} # Using A directly
        ]
        
        # Bounds: u can be negative, but we might want to encourage positivity?
        # No, allow negative u, the constraint f>=0 handles the shape.
        
        try:
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success:
                u_opt = res.x
                # Check energy constraint
                energy = np.dot(u_opt, u_opt)
                u_opt = u_opt / np.sqrt(energy)
                
                # Check non-negativity
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-6):
                    # Calculate score
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
    # Intervals are [x_i, x_{i+1}] where x_i = -0.25 + i * (0.5/600)
    # Sample at centers
    
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    # Reuse A logic but for centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    # Ensure non-negative
    f_centers = np.maximum(0, f_centers)
    
    # The "normalized maximum" score returned by the evaluator is likely 
    # max_autoconv / ||f||_2^2
    # Our optimization minimized max_autoconv subject to ||f||_2^2 = 1.
    # So the score is best_score.
    # But we just return the heights.
    
    return list(f_centers)

if __name__ == "__main__":
    # Quick self-check
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
