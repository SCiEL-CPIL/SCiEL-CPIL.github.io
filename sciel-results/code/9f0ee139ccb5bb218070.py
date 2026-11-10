# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
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

def _nelder_mead_minimize(func, x0, max_iter=5000, tol=1e-12, deadline=None, norm_func=None):
    """
    Nelder-Mead minimization with optional normalization callback.
    """
    n = len(x0)
    
    # Normalize initial point if provided
    if norm_func is not None:
        x0 = norm_func(x0).copy()
    
    # Initial simplex with adaptive step size
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

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere (using quadrature)."""
    f = B_GLOBAL @ c
    norm2_sq = np.sum(W_GLOBAL * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _get_eigenvector_start():
    """
    Get an excellent initial guess by optimizing in the axisymmetric subspace first,
    then using that as a starting point for full optimization.
    """
    dim = 21
    
    # Axisymmetric subspace: only even m modes (m=0, ±2, ±4, ±6, ±8, ±10)
    # Indices: m=0 -> 10, m=±2 -> 8,12, m=±4 -> 6,14, m=±6 -> 4,16, m=±8 -> 2,18, m=±10 -> 0,20
    ax_indices = [10, 8, 12, 6, 14, 4, 16, 2, 18, 0, 20]
    
    # Start with axisymmetric pattern: f(θ) proportional to P_10(cos θ)
    # which is just m=0 mode
    c_ax = np.zeros(dim)
    c_ax[10] = 1.0  # m=0
    
    # Optimize in the axisymmetric subspace using gradient descent
    for _ in range(500):
        f = B_GLOBAL @ c_ax
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
        
        # Project gradient onto axisymmetric subspace
        grad_ax = np.zeros(dim)
        for idx in ax_indices:
            grad_ax[idx] = grad_R[idx]
        
        grad_norm = np.linalg.norm(grad_ax)
        if grad_norm < 1e-14:
            break
        
        # Adaptive step size
        step = 0.05
        improved = False
        for _bt in range(10):
            c_try = c_ax - step * grad_ax
            c_try = _normalize_unit_l2(c_try)
            if _compute_ratio(c_try) < _compute_ratio(c_ax) - 1e-12:
                c_ax = c_try
                improved = True
                break
            step *= 0.5
        
        if not improved:
            break
    
    # Also try a few other axisymmetric patterns
    np.random.seed(12345)
    best_ax = c_ax.copy()
    best_ax_val = _compute_ratio(c_ax)
    
    for trial in range(20):
        c_trial = np.zeros(dim)
        for idx in ax_indices:
            c_trial[idx] = np.random.randn()
        c_trial = _normalize_unit_l2(c_trial)
        
        # Quick gradient descent in axisymmetric subspace
        for _ in range(100):
            f = B_GLOBAL @ c_trial
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
            
            grad_ax = np.zeros(dim)
            for idx in ax_indices:
                grad_ax[idx] = grad_R[idx]
            
            grad_norm = np.linalg.norm(grad_ax)
            if grad_norm < 1e-14:
                break
            
            step = 0.05
            for _bt in range(10):
                c_try = c_trial - step * grad_ax
                c_try = _normalize_unit_l2(c_try)
                if _compute_ratio(c_try) < _compute_ratio(c_trial) - 1e-12:
                    c_trial = c_try
                    break
                step *= 0.5
        
        val = _compute_ratio(c_trial)
        if val < best_ax_val:
            best_ax_val = val
            best_ax = c_trial.copy()
    
    return best_ax

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
        
    # 5. Axisymmetric patterns (exploit rotational symmetry)
    ax_indices = [10, 8, 12, 6, 14, 4, 16, 2, 18, 0, 20]
    
    # Pure m=0 (zonal harmonic)
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    # Mixed even modes with various weightings
    for weights in [
        [1, 1, 1, 1, 1, 1],
        [1, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        [1, 2, 3, 4, 5, 6],
        [1, -1, 1, -1, 1, -1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ]:
        c_pat = np.zeros(dim)
        for idx, w in zip(ax_indices, weights):
            c_pat[idx] = w
        if np.any(c_pat != 0):
            inits.append(c_pat)
    
    # 6. Perturbed eigenvector (axisymmetric start)
    for _ in range(8):
        inits.append(c_eig + 0.1 * np.random.randn(dim))
    
    # 7. Perturbed axisymmetric patterns
    for _ in range(5):
        c_pat = np.zeros(dim)
        for idx in ax_indices:
            c_pat[idx] = np.random.randn()
        inits.append(c_pat)
    
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
    
    # Run Nelder-Mead from the best starts with normalization
    n_starts_to_optimize = 25
    for val, c_start in init_vals[:n_starts_to_optimize]:
        if time.time() > deadline - 15:
            break
        
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_start, max_iter=4000, tol=1e-13, 
                deadline=deadline, norm_func=_normalize_unit_l2
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Additional random restarts if time permits
    n_extra_restarts = 40
    for i in range(n_extra_restarts):
        if time.time() > deadline - 10:
            break
            
        if i % 4 == 0:
            c_rand = np.random.randn(dim)
        elif i % 4 == 1:
            c_rand = best_coeffs + 0.05 * np.random.randn(dim)
        elif i % 4 == 2:
            c_rand = 0.7 * best_coeffs + 0.3 * np.random.randn(dim)
        else:
            # Axisymmetric random
            c_rand = np.zeros(dim)
            ax_indices = [10, 8, 12, 6, 14, 4, 16, 2, 18, 0, 20]
            for idx in ax_indices:
                c_rand[idx] = np.random.randn()
            
        c_rand = _normalize_unit_l2(c_rand)
        
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_rand, max_iter=2500, tol=1e-13,
                deadline=deadline, norm_func=_normalize_unit_l2
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Final polishing: gradient-based refinement with tighter tolerance
    if time.time() < deadline - 20:
        c_polish = best_coeffs.copy()
        for _ in range(1000):
            if time.time() > deadline - 15:
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
            if grad_norm < 1e-15:
                break
            
            # Adaptive step size with backtracking
            step = 0.01
            improved = False
            for _backtrack in range(25):
                c_try = c_polish - step * grad_R
                c_try = _normalize_unit_l2(c_try)
                val_try = objective(c_try)
                if val_try < objective(c_polish) - 1e-16:
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
