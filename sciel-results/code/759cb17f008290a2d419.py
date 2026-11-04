import numpy as np
from scipy.optimize import minimize
import sys

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 40   # Number of harmonics (increased from 25 for more flexibility)
    
    # Grids for optimization
    # x grid for f(x) >= 0 constraint
    # We need enough points to ensure non-negativity is checked well.
    # 600 points is the final resolution, let's use that or slightly more.
    n_x = 600
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    # t grid for autoconvolution peak
    # g(t) is periodic with period L=0.5.
    # We need to resolve the peaks. 1024 is good.
    n_t = 1024
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
    # Note: The autoconvolution of a real even function f(x) = sum u_k psi_k(x)
    # leads to g(t) = sum |c_k|^2 e^{i ...}.
    # The seed code uses: g(t) = 2 * sum u_k^2 B[j,k].
    # Let's verify the normalization.
    # If f is constant sqrt(2), u_0=1, others 0.
    # f(x) = sqrt(2).
    # Autoconvolution h(t) = integral f(x) f(x-t) dx.
    # Since f is constant on [-0.25, 0.25], h(t) is a triangle wave?
    # Wait, the problem says "normalized maximum of its autoconvolution".
    # Usually this means max(h) / ||f||_2^2.
    # ||f||_2^2 = integral f^2 dx = integral 2 dx = 2 * 0.5 = 1.
    # So we minimize max(h).
    # For constant f=sqrt(2), h(t) = integral_{max(-0.25, t-0.25)}^{min(0.25, 0.25-t)} 2 dx.
    # At t=0, h(0) = 2 * 0.5 = 1.
    # The seed code calculates g(t) = 2 * (B @ u_sq).
    # If u_0=1, u_k=0 for k>0, B[j,0]=1.
    # g(t) = 2 * 1 = 2.
    # Why 2?
    # Let's check the definition in the seed code comments.
    # "g(t) = sum c_k^2 e^{i 2 pi k t / L}".
    # If f(x) = sum c_k e^{i 2 pi k x / L}, then autoconvolution is sum |c_k|^2 e^{i 2 pi k t / L}.
    # For constant f=1 on [-0.25, 0.25], c_0 = 1/sqrt(0.5) = sqrt(2).
    # Then g(t) = |c_0|^2 = 2.
    # But the physical autoconvolution at 0 is 1.
    # There is a normalization factor.
    # The task says "normalized maximum".
    # If the seed code returns 2.0 for the constant function, and the score is 1.0,
    # it implies the "seed value" is 2.0? Or does the evaluator divide by something?
    # "score is the upper bound on C1".
    # "search score is seed/value".
    # If seed program returns constant function, its value is V_seed.
    # If V_seed = 2.0 (from the code's internal calculation), then score = 2.0/2.0 = 1.0.
    # So we should minimize the value calculated by the same formula as the seed code.
    # Formula: max(2 * (B @ u_sq)).
    
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def objective_and_gradient(u):
        # u is the vector of coefficients
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        # Smooth max approximation: (sum g_i^p)^(1/p)
        # Using p=20 makes it close to max but differentiable.
        p = 20.0
        # Handle potential negative g_vals? g_vals should be positive if u is real.
        # g_vals = 2 * sum u_k^2 cos(...). Since u_k^2 >= 0, but cos can be negative.
        # However, g(t) is an autoconvolution, so it must be non-negative?
        # No, autoconvolution of a non-negative function is non-negative.
        # So g_vals should be >= 0.
        # To be safe, clip small negatives.
        g_vals = np.maximum(g_vals, 0.0)
        
        # L_p norm
        # To avoid overflow/underflow, we can normalize or just compute directly.
        # With M=40, u_k are small. g_vals are O(1).
        obj = np.sum(g_vals**p) ** (1.0/p)
        
        # Gradient
        # d/d u_k ( sum g_j^p )^(1/p) = (1/p) (sum g_j^p)^(1/p - 1) * sum_j p g_j^(p-1) * dg_j/du_k
        # = (sum g_j^p)^(1/p - 1) * sum_j g_j^(p-1) * dg_j/du_k
        
        # dg_j/du_k = 4 * u_k * B[j, k]
        # Let S = sum_j g_j^(p-1) * B[j, k] for each k.
        # Then grad_k = (sum g_j^p)^(1/p - 1) * 4 * u_k * S_k
        
        # Compute g_j^(p-1)
        g_pow = g_vals**(p-1)
        # S_k = sum_j g_j^(p-1) * B[j, k]
        # S = B.T @ g_pow
        S = B.T @ g_pow
        
        # Base term
        base = (np.sum(g_vals**p))**(1.0/p - 1.0)
        
        grad = 4.0 * base * u * S
        
        return obj, grad

    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        # f(x) = A @ u >= 0
        f_vals = A @ u
        # SLSQP requires g(x) >= 0
        return f_vals

    # Initial guess: Uniform function
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 10
    
    for r in range(n_restarts):
        if r == 0:
            u_start = u0.copy()
        else:
            # Random start on sphere
            u_rand = np.random.randn(M + 1)
            u_start = u_rand / np.linalg.norm(u_rand)
            
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A}
        ]
        
        try:
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            # Check success or just evaluate the result
            u_opt = res.x
            
            # Re-normalize
            energy = np.dot(u_opt, u_opt)
            if energy < 1e-10:
                continue
            u_opt = u_opt / np.sqrt(energy)
            
            # Check non-negativity strictly
            f_vals = A @ u_opt
            if np.all(f_vals >= -1e-6):
                # Calculate raw score using the same formula as seed
                u_sq = u_opt**2
                g_vals = 2.0 * (B @ u_sq)
                score = np.max(g_vals)
                
                # Also check the smoothed objective to see if it's close
                # score_smooth = objective_and_gradient(u_opt)[0]
                
                if score < best_score:
                    best_score = score
                    best_u = u_opt.copy()
                    
        except Exception as e:
            pass
    
    # If optimization failed, use best_u
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
