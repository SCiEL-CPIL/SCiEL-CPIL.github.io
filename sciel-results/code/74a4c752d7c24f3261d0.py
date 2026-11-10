# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, factorial
from scipy.optimize import minimize

# Precompute the real orthonormal spherical harmonics Y_m for degree 10, m=-10..10
# Y_0 = N_0 * P_10^0(cos theta)
# Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m phi) for m > 0
# Y_m = sqrt(2) * N_|m| * P_10^{|m|}(cos theta) * sin(|m| phi) for m < 0
# N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)

def _build_harmonic_basis():
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    m_vals = np.arange(0, 11)
    P = np.zeros((11, n_theta))
    for m in range(11):
        P[m, :] = lpmv(m, 10, x)
    
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0/(4.0*np.pi) * factorial(10-m, exact=False)/factorial(10+m, exact=False))
    
    n_pts = n_theta * n_phi
    B = np.zeros((n_pts, 21))
    
    B[:, 10] = np.tile(N[0] * P[0, :], n_phi)
    
    sqrt2 = np.sqrt(2.0)
    for m in range(1, 11):
        idx_pos = m + 10
        cos_term = np.cos(m * phi)
        val = sqrt2 * N[m] * P[m, :]
        B[:, idx_pos] = (val[:, None] * cos_term[None, :]).ravel()
        
        idx_neg = -m + 10
        sin_term = np.sin(m * phi)
        B[:, idx_neg] = (val[:, None] * sin_term[None, :]).ravel()
    
    weights_phi = np.full(n_phi, 2.0*np.pi/n_phi)
    weights = np.repeat(w, n_phi)
    
    return B, weights, n_pts


B, weights, n_pts = _build_harmonic_basis()

def _compute_ratio(c):
    f = B @ c
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)
    
    if norm2_sq < 1e-30:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    return norm4 / norm2

def _objective_and_grad(c):
    """Compute the L4/L2 ratio and its gradient."""
    f = B @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_4 = np.sum(weights * f4)
    
    if norm2_sq < 1e-30:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = norm4_4 ** 0.25
    ratio = norm4 / norm2
    
    # Gradient of ||f||_4 / ||f||_2
    # d/dc (||f||_4) = (1/4) ||f||_4^{-3} * 4 * sum(weights * f^3 * B^T) = ||f||_4^{-3} * sum(weights * f^3 * B^T)
    # d/dc (||f||_2) = (1/2) ||f||_2^{-1} * 2 * sum(weights * f * B^T) = ||f||_2^{-1} * sum(weights * f * B^T)
    
    d_norm4 = (weights * f**3) @ B / (norm4**3)
    d_norm2 = (weights * f) @ B / norm2
    
    grad = d_norm4 / norm2 - norm4 * d_norm2 / norm2_sq
    
    return ratio, grad

def _normalize_unit_l2(c):
    """Normalize c to have unit L2 norm on the sphere."""
    f = B @ c
    norm2_sq = np.sum(weights * f * f)
    if norm2_sq < 1e-30:
        return c
    return c / np.sqrt(norm2_sq)

def _get_eigenvector_start():
    """Get eigenvector of the quadratic form as initial guess."""
    # The L4 norm squared involves f^4, which is not quadratic.
    # Instead, use the fact that we want to minimize ||f||_4^4 / ||f||_2^4
    # This is equivalent to minimizing ||f||_4^4 subject to ||f||_2^2 = 1
    # 
    # For a simpler approach, use random starts and local optimization.
    return None

def _nelder_mead_minimize(c0, deadline):
    """Custom Nelder-Mead with deadline checking."""
    def objective(c):
        if time.time() > deadline:
            raise TimeoutError()
        r, _ = _objective_and_grad(c)
        return r
    
    try:
        result = minimize(
            objective,
            c0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
        )
        return result.x, result.fun
    except TimeoutError:
        return c0, _compute_ratio(c0)
    except Exception:
        return c0, _compute_ratio(c0)

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    dim = 21
    
    best_c = np.ones(dim, dtype=np.float64)
    best_val = _compute_ratio(best_c)
    
    # Generate diverse initializations
    np.random.seed(42)
    inits = []
    
    # 1. All ones
    inits.append(np.ones(dim))
    
    # 2. Random
    for i in range(20):
        inits.append(np.random.randn(dim))
    
    # 3. Concentrated on single m
    for m_offset in range(dim):
        c_init = np.zeros(dim)
        c_init[m_offset] = 1.0
        inits.append(c_init)
    
    # 4. Pairs (m, -m)
    for m in range(1, 11):
        for sign in [1, -1]:
            c_init = np.zeros(dim)
            c_init[m + 10] = 1.0
            c_init[-m + 10] = sign
            inits.append(c_init)
    
    # 5. Symmetric patterns
    for pattern in range(4):
        c_init = np.zeros(dim)
        c_init[10] = 1.0  # m=0
        for m in range(1, 11):
            if pattern == 0:
                c_init[m + 10] = 1.0
                c_init[-m + 10] = 1.0
            elif pattern == 1:
                c_init[m + 10] = 1.0
                c_init[-m + 10] = -1.0
            elif pattern == 2:
                if m % 2 == 0:
                    c_init[m + 10] = 1.0
                    c_init[-m + 10] = 1.0
                else:
                    c_init[m + 10] = -1.0
                    c_init[-m + 10] = -1.0
            else:
                c_init[m + 10] = (-1)**m
                c_init[-m + 10] = (-1)**m
        inits.append(c_init)
    
    # 6. Sparse random
    for n_nonzero in [2, 3, 5, 7]:
        for _ in range(5):
            c_init = np.zeros(dim)
            indices = np.random.choice(dim, size=n_nonzero, replace=False)
            c_init[indices] = np.random.choice([-1.0, 1.0], size=n_nonzero)
            inits.append(c_init)
    
    # Optimize from each initialization
    for i, c_init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        # Normalize to unit L2
        c_norm = _normalize_unit_l2(c_init)
        
        # Try Nelder-Mead
        c_opt, val = _nelder_mead_minimize(c_norm, deadline - 30.0)
        
        if val < best_val:
            # Validate
            if (len(c_opt) == 21 and 
                not np.any(np.isnan(c_opt)) and 
                not np.any(np.isinf(c_opt)) and
                np.max(np.abs(c_opt)) >= 1e-3 and
                np.max(np.abs(c_opt)) <= 1e5):
                best_val = val
                best_c = c_opt.copy()
    
    # Final refinement with more iterations from best
    if time.time() < deadline - 20.0:
        c_norm = _normalize_unit_l2(best_c)
        c_opt, val = _nelder_mead_minimize(c_norm, deadline - 20.0)
        if val < best_val:
            if (len(c_opt) == 21 and 
                not np.any(np.isnan(c_opt)) and 
                not np.any(np.isinf(c_opt)) and
                np.max(np.abs(c_opt)) >= 1e-3 and
                np.max(np.abs(c_opt)) <= 1e5):
                best_val = val
                best_c = c_opt.copy()
    
    # Final validation
    c_final = best_c
    if len(c_final) != 21:
        c_final = np.ones(21)
    if np.any(np.isnan(c_final)) or np.any(np.isinf(c_final)):
        c_final = np.ones(21)
    max_abs = np.max(np.abs(c_final))
    if max_abs < 1e-3:
        c_final = c_final / max_abs if max_abs > 0 else np.ones(21)
    if np.max(np.abs(c_final)) > 1e5:
        c_final = c_final * (1e5 / np.max(np.abs(c_final)))
    
    return c_final


# EVOLVE-BLOCK-END
