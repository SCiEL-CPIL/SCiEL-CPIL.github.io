import numpy as np
from scipy.optimize import minimize

def construct():
    N = 600
    L = 0.5
    M = 30
    
    n_x = 500
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    n_t = 1000
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute A (f(x) values)
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    # Precompute B (cos terms for g(t))
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def objective_and_gradient(u):
        u_sq = u**2
        g_vals = 2.0 * (B @ u_sq)
        
        # Log-sum-exp for smooth max
        beta = 20.0
        # Shift for numerical stability
        g_max = np.max(g_vals)
        exp_g = np.exp(beta * (g_vals - g_max))
        sum_exp = np.sum(exp_g)
        weights = exp_g / sum_exp
        
        # J = g_max + (1/beta) * log(sum_exp)
        J = g_max + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient
        # dJ/du_k = sum_j w_j * dg_j/du_k
        # dg_j/du_0 = 4 u_0
        # dg_j/du_k = 4 u_k B[j, k]
        # dJ/du = 4 * u * (weights @ B)
        
        W = weights @ B
        grad = 4.0 * u * W
        
        return J, grad

    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        return A @ u

    def constraint_nonneg_grad(u):
        return A

    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    best_score = np.inf
    best_u = u0.copy()
    
    n_restarts = 10
    
    for r in range(n_restarts):
        if r == 0:
            u_start = u0.copy()
        else:
            u_rand = np.random.randn(M + 1)
            u_start = u_rand / np.linalg.norm(u_rand)
            
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': constraint_nonneg_grad}
        ]
        
        try:
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success:
                u_opt = res.x
                energy = np.dot(u_opt, u_opt)
                u_opt = u_opt / np.sqrt(energy)
                
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-4):
                    u_sq = u_opt**2
                    g_vals = 2.0 * (B @ u_sq)
                    score = np.max(g_vals)
                    if score < best_score:
                        best_score = score
                        best_u = u_opt.copy()
                        
        except Exception:
            pass
            
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
