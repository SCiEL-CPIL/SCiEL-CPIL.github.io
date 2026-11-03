# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 195.0

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    # High-resolution grid for matrix extraction
    N = 4000
    x = np.linspace(-15.0, 15.0, N)
    dx = x[1] - x[0]

    # Precompute basis on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)

    # Effective basis vectors (including c12 constraint)
    B0 = H0 - H12 / h12_0
    B4 = H4 - (h4_0 / h12_0) * H12
    B8 = H8 - (h8_0 / h12_0) * H12

    # We need to compute:
    # Q_g(c) = ||g||_2^2
    # Q_x(c) = ||x g||_2^2
    # Q_xi(c) = ||xi hat{g}||_2^2
    
    # Note: For Hermite-Gaussians, ||xi hat{g}||_2^2 = ||x g||_2^2 if g is even?
    # No, ||xi hat{g}||_2^2 = ||x g||_2^2 is only true if g is a Gaussian?
    # Actually, by Parseval, ||xi hat{g}||_2^2 = ||x g||_2^2 is NOT generally true.
    # ||hat{g}||_2 = ||g||_2.
    # ||xi hat{g}||_2^2 = || d/dx g ||_2^2.
    # So we need ||g'||_2^2.
    
    # Let's compute the matrices M_g, M_x, M_d (for derivative)
    # g = c0*B0 + c4*B4 + c8*B8
    # g' = c0*B0' + c4*B4' + c8*B8'
    
    # Numerical derivatives
    B0_p = np.gradient(B0 * gauss, dx) # Wait, g = B*gauss. So g' = (B*gauss)'
    # Actually, it's easier to work with g directly.
    # Let's define the basis functions for g:
    G0 = B0 * gauss
    G4 = B4 * gauss
    G8 = B8 * gauss
    
    G0_p = np.gradient(G0, dx)
    G4_p = np.gradient(G4, dx)
    G8_p = np.gradient(G8, dx)
    
    # Compute norms
    # Q_g = ||g||^2 = sum(g^2) dx
    # Q_x = ||x g||^2 = sum((x*g)^2) dx
    # Q_d = ||g'||^2 = sum((g')^2) dx
    
    # Extract 3x3 matrices
    # Basis vectors in coefficient space: e0=[1,0,0], e4=[0,1,0], e8=[0,0,1]
    # We compute Q(e_i) and Q(e_i+e_j)
    
    def get_Q(func_vals, func_vals_p=None):
        # func_vals: list of arrays [G0, G4, G8]
        # Returns 3x3 matrix M such that c^T M c = sum((c0*G0 + ...)^2) dx
        # If func_vals_p is provided, it's for the derivative term
        pass
        
    # Let's just compute the matrices directly
    # M_g
    M_g = np.zeros((3,3))
    M_x = np.zeros((3,3))
    M_d = np.zeros((3,3))
    
    Gs = [G0, G4, G8]
    Gp = [G0_p, G4_p, G8_p]
    
    # Diagonal terms
    for i in range(3):
        M_g[i,i] = np.sum(Gs[i]**2) * dx
        M_x[i,i] = np.sum((x * Gs[i])**2) * dx
        M_d[i,i] = np.sum(Gp[i]**2) * dx
        
    # Off-diagonal terms
    for i in range(3):
        for j in range(i+1, 3):
            # Q(e_i + e_j) = Q(e_i) + Q(e_j) + 2 M_ij
            # So M_ij = (Q(e_i+e_j) - Q(e_i) - Q(e_j)) / 2
            G_sum = Gs[i] + Gs[j]
            Gp_sum = Gp[i] + Gp[j]
            
            Q_g_sum = np.sum(G_sum**2) * dx
            Q_x_sum = np.sum((x * G_sum)**2) * dx
            Q_d_sum = np.sum(Gp_sum**2) * dx
            
            M_g[i,j] = (Q_g_sum - M_g[i,i] - M_g[j,j]) / 2.0
            M_x[i,j] = (Q_x_sum - M_x[i,i] - M_x[j,j]) / 2.0
            M_d[i,j] = (Q_d_sum - M_d[i,i] - M_d[j,j]) / 2.0
            
            M_g[j,i] = M_g[i,j]
            M_x[j,i] = M_x[i,j]
            M_d[j,i] = M_d[i,j]

    def compute_bound_analytic(c):
        c = np.array(c)
        q_g = c @ M_g @ c
        if q_g < 1e-18:
            return 1e10
        q_x = c @ M_x @ c
        q_d = c @ M_d @ c
        
        # Uncertainty product squared: (Delta x)^2 (Delta xi)^2
        # Delta x^2 = q_x / q_g
        # Delta xi^2 = q_d / q_g
        # Product = q_x * q_d / q_g^2
        return (q_x * q_d) / (q_g * q_g)

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = np.array([c0_init, c4_init, c8_init])
    
    best_score = compute_bound_analytic(seed)
    best_coeffs = seed.copy()

    def time_ok(margin=20.0):
        return (time.time() - start_time) < (deadline - margin)

    # Optimization using the analytic model
    if time_ok(10.0):
        # Multi-start local optimization
        rng = np.random.default_rng(42)
        starts = [seed.copy()]
        for _ in range(20):
            pert = seed + rng.normal(0, 0.1, 3)
            starts.append(pert)
            
        for s in starts:
            if not time_ok(5.0):
                break
            try:
                res = minimize(
                    compute_bound_analytic,
                    s,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x
            except Exception:
                pass

    return best_coeffs

# EVOLVE-BLOCK-END
