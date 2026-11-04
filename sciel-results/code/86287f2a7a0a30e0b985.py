import numpy as np
from scipy.optimize import minimize

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 20   # Number of harmonics (reduced for stability)
    
    # Grids for optimization
    # x grid for f(x) >= 0 constraint
    n_x = 200
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    # t grid for autoconvolution peak
    n_t = 200
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
    # g(t) = 2 * sum_k u_k^2 cos(2 pi k t / L)
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def objective_and_gradient(u):
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        # Smooth max approximation to help gradient descent
        beta = 50.0
        max_g = np.max(g_vals)
        # Use log-sum-exp for smoothness, but for SLSQP, subgradient is often better
        # Let's stick to subgradient for exactness
        idx_max = np.argmax(g_vals)
        
        # Gradient of g_vals[idx] w.r.t u
        # g(idx) = 2 * sum_k u_k^2 B[idx, k]
        # d/du_k g = 4 * u_k * B[idx, k]
        grad = 4.0 * u * B[idx_max, :]
        
        return g_vals[idx_max], grad

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
    
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 3
    
    for r in range(n_restarts):
        if r == 0:
            u_start = u0.copy()
        else:
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
                options={'maxiter': 100, 'ftol': 1e-5}
            )
            
            if res.success:
                u_opt = res.x
                energy = np.dot(u_opt, u_opt)
                if energy > 0:
                    u_opt = u_opt / np.sqrt(energy)
                
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-4):
                    u_sq = u_opt**2
                    g_vals = 2.0 * (B @ u_sq)
                    score = np.max(g_vals)
                    if score < best_score:
                        best_score = score
                        best_u = u_opt.copy()
                        
        except Exception as e:
            pass
    
    # Evaluate on the final 600 step intervals
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    f_centers = np.maximum(0, f_centers)
    
    return list(f_centers)

if __name__ == "__main__":
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
