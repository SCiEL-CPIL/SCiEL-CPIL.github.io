import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s kill

    # Precompute the quadratic form matrices for exact evaluation
    # g(x) = P(x) * exp(-x^2/2) where P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    
    # We need:
    # A = integral(g^2) dx  -> quadratic in c
    # B = integral(x^2 * g^2) dx -> quadratic in c  
    # C = integral(g'^2) dx -> quadratic in c
    # Bound = 4 * B * C / A^2
    
    # Using orthogonality: int H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_mn
    # But we need int x^2 H_m H_n e^{-x^2} and int (d/dx(H_m e^{-x^2/2}))^2 dx
    
    # Let's compute these matrix elements numerically with high precision
    x = np.linspace(-20, 20, 50000)
    dx = x[1] - x[0]
    
    # Hermite polynomials
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 84*x**10 + 2520*x**8 - 30240*x**6 + 166320*x**4 - 302400*x**2 + 10395
    
    basis = [H0(x), H4(x), H8(x), H12(x)]
    n_basis = 4
    
    # Compute A_ij = int H_i H_j e^{-x^2} dx
    A_mat = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            A_mat[i,j] = np.trapz(basis[i] * basis[j] * np.exp(-x**2), x)
    
    # Compute B_ij = int x^2 H_i H_j e^{-x^2} dx
    B_mat = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(n_basis):
            B_mat[i,j] = np.trapz(x**2 * basis[i] * basis[j] * np.exp(-x**2), x)
    
    # Compute C_ij = int (d/dx(H_i e^{-x^2/2})) (d/dx(H_j e^{-x^2/2})) dx
    # d/dx(H_n e^{-x^2/2}) = H_n' e^{-x^2/2} - x H_n e^{-x^2/2}
    # For physicists' Hermite: H_n' = 2n H_{n-1}
    # But we can just compute numerically
    C_mat = np.zeros((n_basis, n_basis))
    derivs = []
    for i in range(n_basis):
        # Numerical derivative
        d = np.gradient(basis[i] * np.exp(-x**2/2), x)
        derivs.append(d)
        for j in range(n_basis):
            C_mat[i,j] = np.trapz(derivs[i] * derivs[j], x)
    
    # Now express in terms of free variables c0, c4, c8
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    # Let c = [c0, c4, c8, c12]
    # c12 = M @ [c0, c4, c8] + b, where M = [-1/10395, -3/10395, -105/10395], b = 0
    
    # So c = G @ v where v = [c0, c4, c8] and
    # G = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1/10395, -3/10395, -105/10395]]
    
    G = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/10395.0, -3.0/10395.0, -105.0/10395.0]
    ])
    
    # Reduced matrices: A_red = G^T A_mat G, etc.
    A_red = G.T @ A_mat @ G
    B_red = G.T @ B_mat @ G
    C_red = G.T @ C_mat @ G
    
    def _bound_and_grad(v):
        """Compute bound and gradient for free variables v = [c0, c4, c8]."""
        c = G @ v
        A_val = c @ A_mat @ c
        B_val = c @ B_mat @ c
        C_val = c @ C_mat @ c
        
        if abs(A_val) < 1e-30:
            return np.inf, np.zeros(3)
        
        bound = 4.0 * B_val * C_val / (A_val ** 2)
        
        # Gradient: d/dv [4 * B * C / A^2]
        # = 4 * (B' * C * A^2 - B * C * 2*A*A') / A^4
        # = 4 * (B' * C - B * C' * (2*A'/A)) / A^2 ... let me do this more carefully
        # f = 4 * B * C * A^{-2}
        # df/dv = 4 * (B' * C * A^{-2} + B * C' * A^{-2} + B * C * (-2) * A^{-3} * A')
        # = 4 * A^{-2} * (B' * C + B * C' - 2 * B * C * A' / A)
        
        # B' = 2 * G^T B_mat c (since B_val = c^T B_mat c, dB/dc = 2 B_mat c)
        # In terms of v: dB/dv = 2 * G^T B_mat c
        dB_dv = 2.0 * (G.T @ B_mat @ c)
        dC_dv = 2.0 * (G.T @ C_mat @ c)
        dA_dv = 2.0 * (G.T @ A_mat @ c)
        
        grad = 4.0 * (A_val ** -2) * (dB_dv * C_val + B_val * dC_dv - 2.0 * B_val * C_val * dA_dv / A_val)
        
        return bound, grad
    
    # Constraint: |g(x)| <= 1 for all x
    # g(x) = (G @ v)^T @ basis(x) * exp(-x^2/2)
    # Let's use a smooth approximation: max_x |g(x)| with soft-max
    
    # Sample points for constraint
    x_con = np.linspace(-10, 10, 2000)
    basis_con = np.array([H0(x_con), H4(x_con), H8(x_con), H12(x_con)])
    gauss_con = np.exp(-x_con**2 / 2.0)
    
    def _constraint(v):
        """Return 1 - max|g(x)|, should be >= 0."""
        c = G @ v
        g_vals = np.array([c[i] * basis_con[i] for i in range(4)]).sum(axis=0) * gauss_con
        return 1.0 - np.max(np.abs(g_vals))
    
    # Objective function for optimizer
    def _objective(v):
        bound, _ = _bound_and_grad(v)
        return bound
    
    def _objective_grad(v):
        bound, grad = _bound_and_grad(v)
        return grad
    
    # Multi-start optimization
    best_bound = np.inf
    best_v = None
    
    starts = [
        [1.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [0.8, 0.0, 0.1],
        [1.2, 0.0, -0.1],
        [1.5, 0.05, 0.0],
        [0.5, -0.05, 0.02],
        [1.0, 0.05, -0.02],
        [-1.13, 0.04, 1.0/3240.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.7, 0.1, 0.05],
        [1.3, -0.05, -0.03],
    ]
    
    for x0 in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            # First check if constraint is satisfied
            if _constraint(x0) < -1e-6:
                # Try to project or skip
                continue
            
            res = minimize(_objective, x0, jac=_objective_grad, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': _constraint}],
                           options={'maxiter': 300, 'ftol': 1e-12})
            
            if res.success or res.fun < best_bound:
                bound_check, _ = _bound_and_grad(res.x)
                if bound_check < best_bound and _constraint(res.x) >= -1e-8:
                    best_bound = bound_check
                    best_v = res.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if best_v is not None and time.time() < deadline - 15:
        try:
            res2 = minimize(_objective, best_v, jac=_objective_grad, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': _constraint}],
                           options={'maxiter': 500, 'ftol': 1e-14})
            bound2, _ = _bound_and_grad(res2.x)
            if bound2 < best_bound and _constraint(res2.x) >= -1e-10:
                best_bound = bound2
                best_v = res2.x.copy()
        except Exception:
            pass
    
    if best_v is None:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    return np.array(best_v)
