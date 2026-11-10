# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    B = np.zeros((n_pts, dim))
    weights = np.repeat(w, n_phi) * (2.0 * np.pi / n_phi)
    
    sqrt2 = np.sqrt(2.0)
    B[:, 10] = np.repeat(N[0] * P[0, :], n_phi)
    
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val_theta[:, None] * cos_term[None, :]).ravel()
        
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        val_theta = sqrt2 * N[m] * P[m, :]
        B[:, idx_neg] = (val_theta[:, None] * sin_term[None, :]).ravel()
        
    return B, weights

B_GLOBAL, W_GLOBAL = _precompute_basis()

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2

def _objective_and_grad(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient for coefficients c.
    Cleaned up gradient computation.
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    if norm4 < 1e-30:
        return np.inf, np.zeros_like(c)
    
    R = norm4 / norm2
    
    # Gradient derivation:
    # N4 = (sum w f^4)^0.25, dN4/dc = N4^-3 * B^T(w f^3)
    # N2 = (sum w f^2)^0.5, dN2/dc = N2^-1 * B^T(w f)
    # R = N4/N2, dR/dc = (1/N2)(N4^-3 G4) - (N4/N2^2)(N2^-1 G2)
    #     = G4/(N2*N4^3) - N4*G2/N2^3
    
    G4 = B_GLOBAL.T @ (W_GLOBAL * f * f2)
    G2 = B_GLOBAL.T @ (W_GLOBAL * f)
    
    grad_R = G4 / (norm2 * norm4**3) - (norm4 * G2) / (norm2**3)
    
    return R, grad_R

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _gradient_descent_polish(c, deadline, max_iter=1000):
    """
    Refined gradient descent with adaptive step size and backtracking.
    Improved with better step size initialization and more robust backtracking.
    """
    c = c.copy()
    current_val = _compute_ratio(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
            
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        # Adaptive step size based on gradient norm
        step = 0.1 / (1.0 + 0.1 * grad_norm)
        improved = False
        for _backtrack in range(60):
            c_try = c - step * grad
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-18:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try alternative: use a smaller fixed step
            step = 0.01
            for _backtrack in range(40):
                c_try = c - step * grad
                c_try = _normalize_unit_l2(c_try)
                val_try = _compute_ratio(c_try)
                if val_try < current_val - 1e-18:
                    c = c_try
                    current_val = val_try
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    return current_val, c

def _lgbfsb_optimize(c0, deadline, norm_func=None):
    """
    Optimize using L-BFGS-B with gradient information.
    """
    def obj_and_grad(c):
        if not np.all(np.isfinite(c)):
            return 1e10, np.zeros_like(c)
        c_norm = norm_func(c)
        val, grad = _objective_and_grad(c_norm)
        if not np.isfinite(val):
            return 1e10, np.zeros_like(c)
        return val, grad
    
    try:
        result = scipy_minimize(
            obj_and_grad,
            c0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxfun': 10000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _generate_symmetric_initial_points(dim):
    """
    Generate initial points based on direct structural analysis of the problem.
    The goal is to find 'flat' functions.
    """
    inits = []
    
    # 1. Zonal (m=0) - Known to have high peaks, but a baseline.
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    # 2. Pure m=10 (High oscillation in phi, low in theta)
    c_m10 = np.zeros(dim)
    c_m10[20] = 1.0 # m=10
    c_m10[0] = 1.0  # m=-10
    inits.append(c_m10)
    
    # 3. All m equal (Isotropic in m-space)
    c_all = np.ones(dim)
    inits.append(c_all)
    
    # 4. Inverse Normalization weights (Equal L2 contribution per mode)
    k = 10
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    c_inv_N = np.zeros(dim)
    c_inv_N[10] = 1.0 / N[0]
    for m in range(1, 11):
        c_inv_N[10+m] = 1.0 / N[m]
        c_inv_N[10-m] = 1.0 / N[m]
    inits.append(c_inv_N)
    
    # 5. Alternating signs to cancel peaks?
    c_alt = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_alt[i] = 1.0 if m % 2 == 0 else -1.0
    inits.append(c_alt)
    
    # 6. Gaussian decay in m (Centered at 0)
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-m**2 / (2 * sigma**2))
        inits.append(c_gauss)
        
    # 7. Gaussian decay in m (Centered at 10)
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-(m-10)**2 / (2 * sigma**2))
        inits.append(c_gauss)

    # 8. Random symmetric
    np.random.seed(42)
    for _ in range(15):
        c_rand = np.random.randn(dim)
        # Enforce symmetry c[10+m] = c[10-m]
        for m in range(1, 11):
            c_rand[10-m] = c_rand[10+m]
        inits.append(c_rand)
        
    # 9. Cosine-weighted modes
    c_cos = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_cos[i] = np.cos(m * np.pi / 20.0)
    inits.append(c_cos)
    
    # 10. Sine-weighted modes
    c_sin = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        c_sin[i] = np.sin(m * np.pi / 20.0)
    inits.append(c_sin)
    
    return inits

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    dim = 21
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    # Generate diverse initial points
    inits = _generate_symmetric_initial_points(dim)
    
    # Evaluate all inits and keep track of best
    init_vals = []
    for c_init in inits:
        c_norm = _normalize_unit_l2(c_init)
        val = objective(c_norm)
        init_vals.append((val, c_norm))
        if val < best_val:
            best_val = val
            best_coeffs = c_norm.copy()
            
    # Sort initializers by value
    init_vals.sort(key=lambda x: x[0])
    
    # Phase 1: L-BFGS-B optimization from best starts
    n_starts_lgbfsb = 25
    for val, c_start in init_vals[:n_starts_lgbfsb]:
        if time.time() > deadline - 150:
            break
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Gradient descent polishing on the best found so far
    if time.time() < deadline - 60:
        c_polish = best_coeffs.copy()
        val_polish, c_polish_opt = _gradient_descent_polish(c_polish, deadline, max_iter=3000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish_opt.copy()
        
        # Additional L-BFGS-B polish with tighter tolerances
        if time.time() < deadline - 30:
            try:
                val_polish2, c_polish2 = _lgbfsb_optimize(
                    best_coeffs, deadline, norm_func=_normalize_unit_l2
                )
                if val_polish2 < best_val:
                    best_val = val_polish2
                    best_coeffs = c_polish2.copy()
            except Exception:
                pass
        
        # Final high-precision gradient descent
        if time.time() < deadline - 10:
            c_final = best_coeffs.copy()
            val_final, c_final_opt = _gradient_descent_polish(c_final, deadline, max_iter=5000)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = c_final_opt.copy()

    # Final validation and normalization
    coeffs = best_coeffs.astype(np.float64)
    
    if not np.all(np.isfinite(coeffs)):
        coeffs = np.ones(dim)
        
    max_c = np.max(np.abs(coeffs))
    
    if max_c < 1e-3:
        if max_c > 0:
            coeffs = coeffs / max_c * 1.0
        else:
            coeffs = np.ones(dim)
    elif max_c > 1e5:
        coeffs = coeffs / max_c * 1.0
        
    return coeffs

# EVOLVE-BLOCK-END
