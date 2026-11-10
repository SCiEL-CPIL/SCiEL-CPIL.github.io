# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize as scipy_minimize
import time

def _precompute_basis():
    """
    Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
    on a 70x130 quadrature grid.
    Returns:
        B: Matrix of shape (n_pts, 21) where B[i, j] = Y_j(theta_i, phi_i)
        weights: Array of shape (n_pts,) with quadrature weights
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
    
    grad_4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))
    grad_2 = 2.0 * (B_GLOBAL.T @ (W_GLOBAL * f))
    
    grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
    
    return R, grad_R

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

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
            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _gradient_descent_polish(c, deadline, max_iter=500):
    """
    Refined gradient descent with adaptive step size and backtracking.
    """
    c = c.copy()
    current_val = _compute_ratio(c)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        val, grad = _objective_and_grad(c)
        if not np.isfinite(val):
            break
            
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        step = 0.01
        improved = False
        for _backtrack in range(40):
            c_try = c - step * grad
            c_try = _normalize_unit_l2(c_try)
            val_try = _compute_ratio(c_try)
            if val_try < current_val - 1e-16:
                c = c_try
                current_val = val_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            break
            
    return current_val, c

def _nelder_mead_minimize(func, x0, max_iter=3000, tol=1e-14, deadline=None, norm_func=None):
    """
    Nelder-Mead minimization with optional normalization callback.
    """
    n = len(x0)
    
    if norm_func is not None:
        x0 = norm_func(x0).copy()
    
    simplex = [x0.copy()]
    base_step = 0.05
    for i in range(n):
        xi = x0.copy()
        xi[i] += base_step
        if norm_func is not None:
            xi = norm_func(xi)
        simplex.append(xi)
    
    fvals = [func(v) for v in simplex]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    best_val = min(fvals)
    best_point = simplex[np.argmin(fvals)].copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol * (1 + abs(fvals[0])):
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + alpha * (centroid - simplex[-1])
        if norm_func is not None:
            xr = norm_func(xr)
        fr = func(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (centroid - simplex[-1])
            if norm_func is not None:
                xe = norm_func(xe)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
            else:
                xc = centroid - rho * (simplex[-1] - centroid)
            if norm_func is not None:
                xc = norm_func(xc)
            fc = func(xc)
            
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    if norm_func is not None:
                        simplex[i] = norm_func(simplex[i])
                    fvals[i] = func(simplex[i])
        
        current_best = min(fvals)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[np.argmin(fvals)].copy()
    
    return best_val, best_point

def _generate_initial_points(dim):
    """Generate diverse initial points for optimization."""
    inits = []
    
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    inits.append(np.ones(dim))
    
    np.random.seed(42)
    for _ in range(20):
        inits.append(np.random.randn(dim))
        
    for i in range(dim):
        c = np.zeros(dim)
        c[i] = 1.0
        inits.append(c)
        
    c_even = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 == 0:
            c_even[i] = 1.0
    inits.append(c_even)
    
    c_odd = np.zeros(dim)
    for i in range(dim):
        m = i - 10
        if m % 2 != 0:
            c_odd[i] = 1.0
    inits.append(c_odd)
    
    c_alt = np.array([1.0 if (i - 10) % 2 == 0 else -1.0 for i in range(dim)])
    inits.append(c_alt)
    
    for bias in [0.5, 0.3, 0.1]:
        c_bias = np.zeros(dim)
        for i in range(dim):
            m = abs(i - 10)
            c_bias[i] = bias ** m
        inits.append(c_bias)
    
    for sigma_m in [1.0, 2.0, 3.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-m**2 / (2 * sigma_m**2))
        inits.append(c_gauss)
    
    for freq in [0.5, 1.0, 1.5, 2.0]:
        c_cos = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_cos[i] = np.cos(freq * m)
        inits.append(c_cos)
    
    for freq in [0.5, 1.0, 1.5, 2.0]:
        c_sin = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_sin[i] = np.sin(freq * m)
        inits.append(c_sin)
    
    return inits

def _eigenvector_start():
    """
    Find a good starting point using eigendecomposition of the Hessian-like matrix.
    The L4/L2 ratio is related to the eigenvalues of a certain matrix.
    """
    dim = 21
    f = B_GLOBAL @ np.ones(dim)
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = norm4_4 ** 0.25
    
    # Build matrix M such that c^T M c approximates the L4 norm
    # M_ij = sum_k w_k Y_i(x_k) Y_j(x_k) Y_k(x_k)^2 / norm2
    # Actually, let's use a simpler approach: 
    # The gradient at c gives direction, but we can also use the Hessian
    
    # Alternative: use power iteration on the "L4 operator"
    c = np.ones(dim)
    c = _normalize_unit_l2(c)
    
    for _ in range(100):
        f = B_GLOBAL @ c
        f2 = f * f
        norm2_sq = np.sum(W_GLOBAL * f2)
        if norm2_sq < 1e-30:
            break
        # "L4 operator" applied to c: sum_k w_k f(x_k)^3 Y_k
        M_c = B_GLOBAL.T @ (W_GLOBAL * f * f2)
        c_new = M_c / (np.linalg.norm(M_c) + 1e-30)
        c_new = _normalize_unit_l2(c_new)
        
        if np.linalg.norm(c_new - c) < 1e-12:
            c = c_new
            break
        c = c_new
    
    return c

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    dim = 21
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    inits = _generate_initial_points(dim)
    
    # Add eigenvector-based start
    c_eig = _eigenvector_start()
    inits.append(c_eig)
    
    init_vals = []
    for c_init in inits:
        c_norm = _normalize_unit_l2(c_init)
        val = objective(c_norm)
        init_vals.append((val, c_norm))
        if val < best_val:
            best_val = val
            best_coeffs = c_norm.copy()
            
    init_vals.sort(key=lambda x: x[0])
    
    # Phase 1: L-BFGS-B optimization from best starts
    n_starts_lgbfsb = 20
    for val, c_start in init_vals[:n_starts_lgbfsb]:
        if time.time() > deadline - 60:
            break
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best starts
    n_starts_nm = 12
    for val, c_start in init_vals[:n_starts_nm]:
        if time.time() > deadline - 60:
            break
        
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_start, max_iter=3000, tol=1e-14, 
                deadline=deadline, norm_func=_normalize_unit_l2
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 3: Additional random restarts with L-BFGS-B
    n_extra_restarts = 40
    for i in range(n_extra_restarts):
        if time.time() > deadline - 40:
            break
            
        if i % 5 == 0:
            c_rand = np.random.randn(dim)
        elif i % 5 == 1:
            c_rand = best_coeffs + 0.05 * np.random.randn(dim)
        elif i % 5 == 2:
            c_rand = 0.7 * best_coeffs + 0.3 * np.random.randn(dim)
        elif i % 5 == 3:
            c_rand = 0.5 * best_coeffs + 0.5 * np.random.randn(dim)
        else:
            c_rand = np.random.randn(dim)
            for i_m in range(dim):
                m = abs(i_m - 10)
                c_rand[i_m] *= 0.5 ** m
            
        c_rand = _normalize_unit_l2(c_rand)
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_rand, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 4: Gradient descent polishing
    if time.time() < deadline - 20:
        c_polish = best_coeffs.copy()
        val_polish, c_polish_opt = _gradient_descent_polish(c_polish, deadline, max_iter=1000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish_opt.copy()
        
        if time.time() < deadline - 10:
            try:
                val_polish2, c_polish2 = _lgbfsb_optimize(
                    best_coeffs, deadline, norm_func=_normalize_unit_l2
                )
                if val_polish2 < best_val:
                    best_val = val_polish2
                    best_coeffs = c_polish2.copy()
            except Exception:
                pass
        
        if time.time() < deadline - 5:
            c_final = best_coeffs.copy()
            val_final, c_final_opt = _gradient_descent_polish(c_final, deadline, max_iter=500)
            if val_final < best_val:
                best_val = val_final
                best_coeffs = c_final_opt.copy()

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
