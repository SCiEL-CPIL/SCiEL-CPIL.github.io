# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial

def rudin_problem10() -> np.ndarray:
    """
    Finds the real coefficients c for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_m c_m Y_m that minimize the ratio ||f||_4 / ||f||_2.
    
    Returns:
        coeffs: np.ndarray of shape (21,)
    """
    k = 10
    
    # 1. Setup Quadrature Grid
    N_theta = 70
    N_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(N_theta)
    
    u = x
    theta = np.arccos(u)
    phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    U, PHI = np.meshgrid(u, phi, indexing='ij')
    
    dphi = 2 * np.pi / N_phi
    W_grid = np.outer(w, np.full(N_phi, dphi))
    W_flat = W_grid.flatten()
    
    # 2. Precompute Spherical Harmonics Y_m(θ, φ)
    m_vals = np.arange(-k, k + 1)
    N_points = N_theta * N_phi
    Y_flat = np.zeros((N_points, 2 * k + 1))
    
    for idx, m in enumerate(m_vals):
        abs_m = abs(m)
        P_val = lpmv(abs_m, k, U)
        
        # Calculate normalization constant N_m
        # N_m = sqrt((2k+1)/(4pi) * (k-m)!/(k+m)!)
        log_num = np.log(2 * k + 1) - np.log(4 * np.pi) + np.log(factorial(k - abs_m)) - np.log(factorial(k + abs_m))
        N_m = np.sqrt(np.exp(log_num))
        
        if m == 0:
            Y_val = N_m * P_val
        elif m > 0:
            Y_val = np.sqrt(2) * N_m * P_val * np.cos(m * PHI)
        else:
            Y_val = np.sqrt(2) * N_m * P_val * np.sin(abs_m * PHI)
        
        Y_flat[:, idx] = Y_val.flatten()
    
    # 3. Define Objective, Gradient, and Hessian
    # We minimize ||f||_4^4 subject to ||f||_2^2 = 1
    def objective(c):
        f = Y_flat @ c
        return np.sum(W_flat * (f ** 4))
    
    def gradient(c):
        f = Y_flat @ c
        # d/dc sum w_i f_i^4 = sum w_i 4 f_i^3 (Y_i)
        return 4.0 * (Y_flat.T @ (W_flat * (f ** 3)))
    
    def hessian(c):
        f = Y_flat @ c
        # d2/dc2 sum w_i f_i^4 = sum w_i 12 f_i^2 Y_i Y_i^T
        w_f2 = W_flat * (f ** 2)
        # Efficient computation: (Y * w_f2[:, None]).T @ Y
        return 12.0 * ((Y_flat * w_f2[:, None]).T @ Y_flat)
    
    # 4. Optimization using trust-constr
    best_c = None
    best_val = np.inf
    
    inits = []
    
    # Structured initializations
    # 1. All ones
    c0 = np.ones(2 * k + 1)
    c0 /= np.linalg.norm(c0)
    inits.append(c0)
    
    # 2. Single modes
    for idx in range(2 * k + 1):
        c_single = np.zeros(2 * k + 1)
        c_single[idx] = 1.0
        inits.append(c_single)
        
    # 3. Even m modes
    c_even = np.zeros(2 * k + 1)
    for i, m in enumerate(m_vals):
        if m % 2 == 0:
            c_even[i] = 1.0
    if np.linalg.norm(c_even) > 0:
        c_even /= np.linalg.norm(c_even)
        inits.append(c_even)
        
    # 4. Odd m modes
    c_odd = np.zeros(2 * k + 1)
    for i, m in enumerate(m_vals):
        if m % 2 != 0:
            c_odd[i] = 1.0
    if np.linalg.norm(c_odd) > 0:
        c_odd /= np.linalg.norm(c_odd)
        inits.append(c_odd)
        
    # 5. Alternating signs
    c_alt = np.zeros(2 * k + 1)
    for i, m in enumerate(m_vals):
        c_alt[i] = 1.0 if m % 2 == 0 else -1.0
    c_alt /= np.linalg.norm(c_alt)
    inits.append(c_alt)
    
    # 6. Random initializations
    rng = np.random.default_rng(12345)
    for _ in range(15):
        cr = rng.standard_normal(2 * k + 1)
        cr /= np.linalg.norm(cr)
        inits.append(cr)
        
    # 7. Gaussian-like distribution over m
    c_gauss = np.zeros(2 * k + 1)
    for i, m in enumerate(m_vals):
        c_gauss[i] = np.exp(-m**2 / 20.0)
    c_gauss /= np.linalg.norm(c_gauss)
    inits.append(c_gauss)
    
    for c_init in inits:
        try:
            res = minimize(
                objective, 
                c_init, 
                method='trust-constr',
                jac=gradient,
                hess=hessian,
                constraints=[{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2 * c}],
                options={'maxiter': 500, 'gtol': 1e-10, 'xtol': 1e-10}
            )
            
            if hasattr(res, 'fun') and res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except Exception:
            pass
            
    if best_c is None:
        best_c = np.ones(2 * k + 1)
        
    best_c = best_c / np.linalg.norm(best_c)
    
    return best_c

# EVOLVE-BLOCK-END
