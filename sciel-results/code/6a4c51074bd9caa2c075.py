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
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
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

def _gradient_descent_polish(c, deadline, max_iter=2000):
    """
    Refined gradient descent with adaptive step size and backtracking.
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
        
        step = 0.1
        improved = False
        for _backtrack in range(50):
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
            if step < 1e-15:
                break
            step = 0.5
            for _backtrack in range(30):
                c_try = c + step * grad
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
        if deadline is not None and time.time() > deadline - 2:
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

def _differential_evolution_search(c0, deadline, norm_func=None, n_pop=40, max_iter=100):
    """
    Differential evolution for global search.
    """
    dim = len(c0)
    np.random.seed(42)
    
    # Initialize population
    pop = [c0.copy()]
    for _ in range(n_pop - 1):
        c_rand = np.random.randn(dim)
        c_rand = norm_func(c_rand)
        pop.append(c_rand)
    
    # Evaluate initial population
    pop_vals = [norm_func(c) for c in pop]
    pop_vals = [(_compute_ratio(c), c) for c in pop_vals]
    pop_vals.sort(key=lambda x: x[0])
    
    best_val = pop_vals[0][0]
    best_c = pop_vals[0][1].copy()
    
    F = 0.8
    CR = 0.9
    
    for it in range(max_iter):
        if time.time() > deadline - 60:
            break
        
        new_pop = [pop_vals[0][1].copy()]
        new_vals = [pop_vals[0][0]]
        
        for i in range(1, n_pop):
            # Select three distinct random individuals
            a, b, c = np.random.choice(n_pop, 3, replace=False)
            
            # Mutation
            mutant = pop_vals[a][1] + F * (pop_vals[b][1] - pop_vals[c][1])
            mutant = norm_func(mutant)
            
            # Crossover
            trial = mutant.copy()
            for j in range(dim):
                if np.random.rand() < CR or j == np.random.randint(dim):
                    trial[j] = mutant[j]
            
            trial = norm_func(trial)
            trial_val = _compute_ratio(trial)
            
            if trial_val < pop_vals[i][0]:
                new_pop.append(trial)
                new_vals.append(trial_val)
            else:
                new_pop.append(pop_vals[i][1].copy())
                new_vals.append(pop_vals[i][0])
        
        # Sort new population
        new_pop_sorted = sorted(zip(new_vals, new_pop), key=lambda x: x[0])
        pop_vals = [(v, c) for v, c in new_pop_sorted]
        
        if pop_vals[0][0] < best_val:
            best_val = pop_vals[0][0]
            best_c = pop_vals[0][1].copy()
    
    return best_val, best_c

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
            options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 5000}
        )
        if result.success or result.fun < 1e10:
            c_opt = norm_func(result.x)
            return result.fun, c_opt
    except Exception:
        pass
    
    return np.inf, c0

def _generate_initial_points(dim, n_random=50):
    """Generate diverse initial points for optimization."""
    inits = []
    
    # 1. Zonal harmonic (m=0)
    c_zonal = np.zeros(dim)
    c_zonal[10] = 1.0
    inits.append(c_zonal)
    
    # 2. All ones
    inits.append(np.ones(dim))
    
    # 3. Sparse initializations (single m)
    for i in range(dim):
        c = np.zeros(dim)
        c[i] = 1.0
        inits.append(c)
        
    # 4. Symmetric patterns
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
    
    # 5. Low-m bias
    for bias in [0.5, 0.3, 0.1]:
        c_bias = np.zeros(dim)
        for i in range(dim):
            m = abs(i - 10)
            c_bias[i] = bias ** m
        inits.append(c_bias)
    
    # 6. Gaussian decay in m
    for sigma_m in [1.0, 2.0, 3.0]:
        c_gauss = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_gauss[i] = np.exp(-m**2 / (2 * sigma_m**2))
        inits.append(c_gauss)
    
    # 7. Cosine patterns in m
    for freq in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]:
        c_cos = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_cos[i] = np.cos(freq * m)
        inits.append(c_cos)
    
    # 8. Sine patterns in m
    for freq in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]:
        c_sin = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_sin[i] = np.sin(freq * m)
        inits.append(c_sin)
    
    # 9. Random normal with various seeds
    for seed in range(n_random):
        np.random.seed(seed * 12345 + 67890)
        inits.append(np.random.randn(dim))
        
    # 10. Low-m biased random
    for seed in range(20):
        np.random.seed(seed * 98765 + 43210)
        c_rand = np.random.randn(dim)
        for i in range(dim):
            m = abs(i - 10)
            c_rand[i] *= 0.5 ** m
        inits.append(c_rand)
        
    # 11. High-m biased random
    for seed in range(10):
        np.random.seed(seed * 54321 + 13579)
        c_rand = np.random.randn(dim)
        for i in range(dim):
            m = abs(i - 10)
            c_rand[i] *= 0.7 ** m
        inits.append(c_rand)
        
    # 12. Sparse random (only a few non-zero)
    for seed in range(20):
        np.random.seed(seed * 24680 + 13579)
        c_sparse = np.zeros(dim)
        n_nonzero = np.random.randint(1, 6)
        indices = np.random.choice(dim, n_nonzero, replace=False)
        c_sparse[indices] = np.random.randn(n_nonzero)
        inits.append(c_sparse)
        
    # 13. Block patterns
    for block_size in [2, 3, 4, 5]:
        for start in range(0, dim - block_size + 1, max(1, block_size - 1)):
            c_block = np.zeros(dim)
            c_block[start:start+block_size] = 1.0
            inits.append(c_block)
            
    # 14. Alternating with varying amplitude
    for amp_decay in [0.5, 0.7, 0.9]:
        c_alt = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            sign = 1.0 if m % 2 == 0 else -1.0
            c_alt[i] = sign * (amp_decay ** abs(m))
        inits.append(c_alt)
        
    # 15. Two-peaked patterns
    for p1 in range(-10, 11, 3):
        for p2 in range(-10, 11, 3):
            if p1 != p2:
                c_two = np.zeros(dim)
                for i in range(dim):
                    m = i - 10
                    c_two[i] = np.exp(-(m-p1)**2) + np.exp(-(m-p2)**2)
                inits.append(c_two)
                
    # 16. Polynomial in m
    for coeffs in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        c_poly = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_poly[i] = coeffs[0] + coeffs[1] * m + coeffs[2] * m**2
        inits.append(c_poly)
        
    # 17. Complex-like patterns (real part of exp(i*freq*m))
    for freq in np.arange(0.1, 3.0, 0.1):
        c_complex = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_complex[i] = np.cos(freq * m + 0.5)
        inits.append(c_complex)
        
    # 18. Banded patterns (non-zero in a band around 0)
    for band_width in [3, 5, 7, 9, 11, 15, 21]:
        c_band = np.zeros(dim)
        center = 10
        half_width = band_width // 2
        start = max(0, center - half_width)
        end = min(dim, center + half_width + 1)
        c_band[start:end] = 1.0
        inits.append(c_band)
        
    # 19. Weighted sum of zonal and tesseral
    for w_tess in np.arange(0.1, 1.5, 0.1):
        c_mix = np.zeros(dim)
        c_mix[10] = 1.0  # zonal
        for m in [1, 2, 3, 4, 5]:
            c_mix[10+m] = w_tess
            c_mix[10-m] = w_tess
        inits.append(c_mix)
        
    # 20. Phase-shifted patterns
    for phase in np.arange(0, 2*np.pi, 0.5):
        c_phase = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_phase[i] = np.cos(m * 0.5 + phase)
        inits.append(c_phase)
        
    # 21. Antisymmetric patterns (odd functions in m)
    for amp in [0.5, 0.7, 0.9]:
        c_anti = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_anti[i] = np.sign(m) * (amp ** abs(m)) if m != 0 else 0
        inits.append(c_anti)
        
    # 22. Quadratic phase patterns
    for a in [0.1, 0.2, 0.3, 0.5]:
        c_quad = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_quad[i] = np.cos(a * m**2)
        inits.append(c_quad)
        
    # 23. Mixed frequency patterns
    for f1, f2 in [(0.5, 1.5), (1.0, 2.0), (0.3, 0.7), (1.5, 2.5)]:
        c_mix_freq = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            c_mix_freq[i] = np.cos(f1 * m) + 0.5 * np.sin(f2 * m)
        inits.append(c_mix_freq)
        
    # 24. Axisymmetric (even m only) with varying zonal weight
    for zonal_w in np.linspace(0.5, 2.0, 16):
        c_ax = np.zeros(dim)
        c_ax[10] = zonal_w
        for m in [2, 4, 6, 8, 10]:
            c_ax[10+m] = 1.0
            c_ax[10-m] = 1.0
        inits.append(c_ax)
        
    # 25. Axisymmetric with quadratic decay in |m|
    for decay in [0.3, 0.5, 0.7, 0.9]:
        c_ax2 = np.zeros(dim)
        for i in range(dim):
            m = i - 10
            if m % 2 == 0:
                c_ax2[i] = decay ** (abs(m) // 2)
        inits.append(c_ax2)
        
    return inits

def _axisymmetric_search(deadline):
    """
    Exploit rotational symmetry: search over axisymmetric (even-m) solutions.
    The L4/L2 ratio is invariant under rotation, so we can restrict to
    solutions where f is axisymmetric (only even m modes, with c_{-m} = c_m).
    This reduces the effective dimension from 21 to 11 (m = 0, 2, 4, 6, 8, 10).
    """
    dim = 21
    even_m_indices = [10 + m for m in [0, 2, 4, 6, 8, 10]]  # indices for m=0,2,4,6,8,10
    n_params = len(even_m_indices)
    
    def full_from_params(params):
        """Map 6 parameters to full 21-dim coefficient vector (axisymmetric)."""
        c = np.zeros(dim)
        for j, idx in enumerate(even_m_indices):
            m = idx - 10
            if m == 0:
                c[idx] = params[j]
            else:
                c[idx] = params[j]
                c[10 - m] = params[j]  # c_{-m} = c_m for axisymmetric
        return c
    
    def objective_params(params):
        c = full_from_params(params)
        c = _normalize_unit_l2(c)
        return _compute_ratio(c)
    
    # Grid search over the 6D parameter space
    best_val = np.inf
    best_params = None
    
    # Start with zonal-only
    p0 = np.zeros(n_params)
    p0[0] = 1.0
    v0 = objective_params(p0)
    if v0 < best_val:
        best_val = v0
        best_params = p0.copy()
    
    # Try adding higher modes
    for trial in range(200):
        if time.time() > deadline - 60:
            break
        np.random.seed(trial * 31415 + 27182)
        params = np.random.randn(n_params) * 0.5
        params[0] += 1.0  # bias toward zonal
        v = objective_params(params)
        if v < best_val:
            best_val = v
            best_params = params.copy()
    
    # Local optimization from best
    if best_params is not None and time.time() < deadline - 30:
        try:
            result = scipy_minimize(
                objective_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-18}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    c_best = full_from_params(best_params)
    c_best = _normalize_unit_l2(c_best)
    return best_val, c_best

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
    
    # Phase 0: Axisymmetric search (exploit rotational symmetry)
    if time.time() < deadline - 120:
        try:
            val_ax, c_ax = _axisymmetric_search(deadline - 60)
            if val_ax < best_val:
                best_val = val_ax
                best_coeffs = c_ax.copy()
        except Exception:
            pass
    
    # Generate diverse initial points
    inits = _generate_initial_points(dim, n_random=50)
    
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
    n_starts_lgbfsb = 30
    for val, c_start in init_vals[:n_starts_lgbfsb]:
        if time.time() > deadline - 120:
            break
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_start, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best starts (for robustness)
    n_starts_nm = 15
    for val, c_start in init_vals[:n_starts_nm]:
        if time.time() > deadline - 120:
            break
        
        try:
            val_opt, c_opt = _nelder_mead_minimize(
                objective, c_start, max_iter=2000, tol=1e-15, 
                deadline=deadline, norm_func=_normalize_unit_l2
            )
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass

    # Phase 3: Additional random restarts with L-BFGS-B
    n_extra_restarts = 50
    for i in range(n_extra_restarts):
        if time.time() > deadline - 60:
            break
            
        if i % 5 == 0:
            np.random.seed(i * 777 + 12345)
            c_rand = np.random.randn(dim)
        elif i % 5 == 1:
            c_rand = best_coeffs + 0.05 * np.random.randn(dim)
        elif i % 5 == 2:
            c_rand = 0.8 * best_coeffs + 0.2 * np.random.randn(dim)
        elif i % 5 == 3:
            # Low-m biased random
            c_rand = np.random.randn(dim)
            for i_m in range(dim):
                m = abs(i_m - 10)
                c_rand[i_m] *= 0.5 ** m
        else:
            # Sparse random
            c_rand = np.zeros(dim)
            n_nonzero = np.random.randint(1, 8)
            indices = np.random.choice(dim, n_nonzero, replace=False)
            c_rand[indices] = np.random.randn(n_nonzero)
            
        c_rand = _normalize_unit_l2(c_rand)
        
        try:
            val_opt, c_opt = _lgbfsb_optimize(c_rand, deadline, norm_func=_normalize_unit_l2)
            
            if val_opt < best_val:
                best_val = val_opt
                best_coeffs = c_opt.copy()
        except Exception:
            pass
    
    # Phase 3.5: Differential evolution for global search
    if time.time() < deadline - 90:
        try:
            val_de, c_de = _differential_evolution_search(
                best_coeffs, deadline, norm_func=_normalize_unit_l2, 
                n_pop=30, max_iter=80
            )
            if val_de < best_val:
                best_val = val_de
                best_coeffs = c_de.copy()
        except Exception:
            pass

    # Phase 4: Gradient descent polishing
    if time.time() < deadline - 30:
        c_polish = best_coeffs.copy()
        val_polish, c_polish_opt = _gradient_descent_polish(c_polish, deadline, max_iter=2000)
        if val_polish < best_val:
            best_val = val_polish
            best_coeffs = c_polish_opt.copy()
        
        # High-precision L-BFGS-B polish with tighter tolerances
        if time.time() < deadline - 20:
            try:
                def obj_and_grad_hp(c):
                    if not np.all(np.isfinite(c)):
                        return 1e10, np.zeros_like(c)
                    c_norm = _normalize_unit_l2(c)
                    val, grad = _objective_and_grad(c_norm)
                    if not np.isfinite(val):
                        return 1e10, np.zeros_like(c)
                    return val, grad
                
                result_hp = scipy_minimize(
                    obj_and_grad_hp,
                    best_coeffs,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxfun': 10000}
                )
                if result_hp.success or result_hp.fun < 1e10:
                    c_hp = _normalize_unit_l2(result_hp.x)
                    if result_hp.fun < best_val:
                        best_val = result_hp.fun
                        best_coeffs = c_hp.copy()
            except Exception:
                pass
        
        # Final gradient descent
        if time.time() < deadline - 5:
            c_final = best_coeffs.copy()
            val_final, c_final_opt = _gradient_descent_polish(c_final, deadline, max_iter=1000)
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
