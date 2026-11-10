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
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    # Phi points
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Total points
    n_pts = n_theta * n_phi
    dim = 2 * k + 1
    
    # Precompute P_10^m(x) for m=0..10
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
        
    # Normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        num = 1.0
        for i in range(1, k - m + 1):
            num *= i
        den = 1.0
        for i in range(1, k + m + 1):
            den *= i
        N[m] = np.sqrt((2 * k + 1) / (4 * np.pi) * (num / den))
        
    # Build B matrix
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

# Precompute at module load time
B_GLOBAL, W_GLOBAL = _precompute_basis()

def _compute_ratio(c):
    """
    Compute R = ||f||_4 / ||f||_2 for coefficients c.
    c has shape (21,).
    """
    f = B_GLOBAL @ c
    f2 = f * f
    norm2_sq = np.sum(W_GLOBAL * f2)
    
    if norm2_sq < 1e-30:
        return np.inf
        
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.sum(W_GLOBAL * f2 * f2)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2

def _objective_and_grad(c):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient for coefficients c.
    Returns (value, gradient).
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
    
    # Gradient of R w.r.t. c
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
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 10000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _nelder_mead_minimize(func, x0, max_iter=5000, tol=1e-12, deadline=None, norm_func=None):
    """
    Nelder-Mead minimization with optional normalization callback.
    """
    n = len(x0)
    
    if norm_func is not None:
        x0 = norm_func(x0).copy()
    
    simplex = [x0.copy()]
    base_step = 0.1
    for i in range(n):
        xi = x0.copy()
        step = base_step if abs(x0[i]) > 1e-6 else base_step
        xi[i] += step
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

def _get_eigenvector_start():
    """
    Solve the generalized eigenvalue problem to get an excellent initial guess.
    """
    dim = 21
    
    np.random.seed(12345)
    c = np.random.randn(dim)
    c = _normalize_unit_l2(c)
    
    for _ in range(200):
        f = B_GLOBAL @ c
        f2 = f * f
        norm2_sq = np.sum(W_GLOBAL * f2)
        if norm2_sq < 1e-30:
            break
        
        grad_4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))
        grad_2 = 2.0 * (B_GLOBAL.T @ (W_GLOBAL * f))
        
        norm4_4 = np.sum(W_GLOBAL * f2 * f2)
        if norm4_4 < 1e-30:
            break
        norm4 = norm4_4 ** 0.25
        norm2 = np.sqrt(norm2_sq)
        
        grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
        
        grad_norm = np.linalg.norm(grad_R)
        if grad_norm < 1e-15:
            break
        
        step = 0.01
        c_new = c - step * grad_R
        c_new = _normalize_unit_l2(c_new)
        
        if _compute_ratio(c_new) < _compute_ratio(c):
            c = c_new
        else:
            step *= 0.5
            c_new = c - step * grad_R
            c_new = _normalize_unit_l2(c_new)
            if _compute_ratio(c_new) < _compute_ratio(c):
                c = c_new
    
    return c

def _axisymmetric_search(c0, deadline):
    """
    Search in the subspace where c_{-m} = 0 for m=1..10.
    This reduces dimensionality from 21 to 11.
    """
    dim_reduced = 11
    dim_full = 21
    
    def map_reduced_to_full(u):
        c = np.zeros(dim_full)
        c[10] = u[0]
        for m in range(1, 11):
            c[10 + m] = u[m]
            # c[10 - m] remains 0
        return c

    def map_full_to_reduced(c):
        u = np.zeros(dim_reduced)
        u[0] = c[10]
        for m in range(1, 11):
            u[m] = c[10 + m]
        return u

    u0 = map_full_to_reduced(c0)
    
    def obj_and_grad_u(u):
        if not np.all(np.isfinite(u)):
            return 1e10, np.zeros_like(u)
        c = map_reduced_to_full(u)
        c_norm = _normalize_unit_l2(c)
        val, grad_c = _objective_and_grad(c_norm)
        
        # Project gradient back to reduced space
        # grad_u = J^T grad_c where J is the embedding matrix
        # Here, J is sparse.
        grad_u = np.zeros(dim_reduced)
        grad_u[0] = grad_c[10]
        for m in range(1, 11):
            grad_u[m] = grad_c[10 + m]
            
        if not np.isfinite(val):
            return 1e10, np.zeros_like(u)
        return val, grad_u

    try:
        result = scipy_minimize(
            obj_and_grad_u,
            u0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 10000}
        )
        if result.success or result.fun < 1e10:
            u_opt = result.x
            c_opt = map_reduced_to_full(u_opt)
            c_opt = _normalize_unit_l2(c_opt)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients c_m for m = -10..10 minimizing L4/L2 ratio of degree-10 spherical harmonic.
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    dim = 21
    
    def objective(c):
        if not np.all(np.isfinite(c)):
            return 1e10
        return _compute_ratio(c)
    
    best_coeffs = None
    best_val = np.inf
    
    # Get excellent starting point from eigenvector/power iteration
    c_eig = _get_eigenvector_start()
    val_eig = objective(c_eig)
    best_val = val_eig
    best_coeffs = c_eig.copy()
    
    # Generate diverse initial points
    inits = []
    
    # 1. The eigenvector start (best)
    inits.append(c_eig.copy())
    
    # 2. All ones
    inits.append(np.ones(dim))
    
    # 3. Random normal
    np.random.seed(42)
    for _ in range(15):
        inits.append(np.random.randn(dim))
        
    # 4. Sparse initializations (single m)
    for i in range(dim):
        c = np.zeros(dim)
        c[i] = 1.0
        inits.append(c)
        
    # 5. Symmetric patterns
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
    
    # 6. Alternating signs
    c_alt = np.array([1.0 if (i - 10) % 2 == 0 else -1.0 for i in range(dim)])
    inits.append(c_alt)
    
    # 7. Perturbed eigenvector
    for _ in range(5):
        inits.append(c_eig + 0.1 * np.random.randn(dim))
    
    # Evaluate all inits
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
    
    # Phase 1: L-BFGS-B optimization from best starts (fast, gradient-based)
    n_starts_lgbfsb = 10
    for val, c_start in init_vals[:n_starts_lgbfsb]:
        if time.time() > deadline - 30:
            break
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best starts (for robustness)
    n_starts_nm = 8
    for val, c_start in init_vals[:n_starts_nm]:
        if time.time() > deadline - 30:
            break
        
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_start, max_iter=2000, tol=1e-12, 
                deadline=deadline, norm_func=_normalize_unit_l2
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Axisymmetric Search (Reduced Dimensionality)
    # Try the best full-dimensional solution as a start for the reduced search
    if time.time() < deadline - 30:
        try:
            val_ax, c_ax = _axisymmetric_search(best_coeffs, deadline)
            if val_ax < best_val:
                best_val = val_ax
                best_coeffs = c_ax.copy()
        except Exception:
            pass
            
    # Also try axisymmetric search from random starts in reduced space
    for i in range(5):
        if time.time() > deadline - 20:
            break
        u_rand = np.random.randn(11)
        c_rand_full = np.zeros(21)
        c_rand_full[10] = u_rand[0]
        for m in range(1, 11):
            c_rand_full[10+m] = u_rand[m]
        c_rand_full = _normalize_unit_l2(c_rand_full)
        
        try:
            val_ax, c_ax = _axisymmetric_search(c_rand_full, deadline)
            if val_ax < best_val:
                best_val = val_ax
                best_coeffs = c_ax.copy()
        except Exception:
            pass

    # Phase 4: Additional random restarts with L-BFGS-B
    n_extra_restarts = 20
    for i in range(n_extra_restarts):
        if time.time() > deadline - 20:
            break
            
        if i % 3 == 0:
            c_rand = np.random.randn(dim)
        elif i % 3 == 1:
            c_rand = best_coeffs + 0.05 * np.random.randn(dim)
        else:
            c_rand = 0.7 * best_coeffs + 0.3 * np.random.randn(dim)
            
        c_rand = _normalize_unit_l2(c_rand)
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_rand, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Final polishing: high-precision gradient-based refinement with L-BFGS-B
    if time.time() < deadline - 10:
        c_polish = best_coeffs.copy()
        try:
            val_polish, c_polish_opt = _lgbfsb_optimize(
                c_polish, deadline, norm_func=_normalize_unit_l2
            )
            if val_polish < best_val:
                best_val = val_polish
                best_coeffs = c_polish_opt.copy()
        except Exception:
            pass
        
        # Additional manual gradient descent polish
        c_polish = best_coeffs.copy()
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            f = B_GLOBAL @ c_polish
            f2 = f * f
            norm2_sq = np.sum(W_GLOBAL * f2)
            if norm2_sq < 1e-30:
                break
            
            norm4_4 = np.sum(W_GLOBAL * f2 * f2)
            if norm4_4 < 1e-30:
                break
            norm4 = norm4_4 ** 0.25
            norm2 = np.sqrt(norm2_sq)
            
            grad_4 = 4.0 * (B_GLOBAL.T @ (W_GLOBAL * f * f2))
            grad_2 = 2.0 * (B_GLOBAL.T @ (W_GLOBAL * f))
            
            grad_R = (1.0 / (4.0 * norm4**3 * norm2)) * grad_4 - (norm4 / (2.0 * norm2**3)) * grad_2
            
            grad_norm = np.linalg.norm(grad_R)
            if grad_norm < 1e-14:
                break
            
            # Adaptive step size with backtracking
            step = 0.005
            improved = False
            for _backtrack in range(25):
                c_try = c_polish - step * grad_R
                c_try = _normalize_unit_l2(c_try)
                val_try = objective(c_try)
                if val_try < objective(c_polish) - 1e-15:
                    c_polish = c_try
                    improved = True
                    break
                step *= 0.5
            
            if not improved:
                break
        
        if objective(c_polish) < best_val:
            best_val = objective(c_polish)
            best_coeffs = c_polish.copy()

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
