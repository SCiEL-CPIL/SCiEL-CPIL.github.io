"""Improved growth-index approximation with precomputed ODE reference grid."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL."""
    a = np.maximum(a, 1e-10)
    Omega_DE0 = 1.0 - Om0
    # rho_DE(a)/rho_DE(1) = a^(-3(1+w0+wa)) * exp(3*wa*(a-1))
    ratio_DE = (a ** (-3.0*(1.0+w0+wa))) * np.exp(3.0*wa*(a-1.0))
    Om_a = Om0 * a**(-3.0) / (Om0 * a**(-3.0) + Omega_DE0 * ratio_DE)
    return Om_a

def _solve_growth_grid(a_grid, Om0, w0, wa, n_steps=2000):
    """Solve growth ODE on a grid using RK4. Returns D_norm, f on grid."""
    # ODE: D'' + (3/2)Om(a)D' - (3/2)Om(a)D = 0, primes d/dlna
    # In terms of x=ln(a): let a=exp(x), D(x), dD/dx = a*D_a
    # d²D/dx² = a*D_a + a²*D_aa
    # Equation becomes: (a*D_a + a²*D_aa) + (3/2)Om*a*D_a - (3/2)Om*D = 0
    # Simpler: work directly in x. Let F = dD/dx. Then dF/dx + (3/2)Om(a(x))*F - (3/2)Om(a(x))*D = 0
    # dD/dx = F
    
    x_grid = np.log(a_grid)
    n = len(x_grid)
    dx = x_grid[1] - x_grid[0] if n > 1 else 0.01
    
    D = np.zeros(n)
    F = np.zeros(n)  # F = dD/dx
    
    # Initial conditions at early times: matter domination, D ~ a ~ exp(x), F ~ 1
    D[0] = a_grid[0]
    F[0] = 1.0
    
    for i in range(n - 1):
        x_i = x_grid[i]
        a_i = np.exp(x_i)
        Om_i = _Om_of_a(a_i, Om0, w0, wa)
        
        def rhs(D_val, F_val, a_val):
            Om_val = _Om_of_a(a_val, Om0, w0, wa)
            dD = F_val
            dF = -(1.5 * Om_val) * F_val + (1.5 * Om_val) * D_val
            return dD, dF
        
        k1_D, k1_F = rhs(D[i], F[i], a_i)
        a_m = np.exp(x_i + 0.5*dx)
        k2_D, k2_F = rhs(D[i] + 0.5*dx*k1_D, F[i] + 0.5*dx*k1_F, a_m)
        k3_D, k3_F = rhs(D[i] + 0.5*dx*k2_D, F[i] + 0.5*dx*k2_F, a_m)
        a_e = np.exp(x_i + dx)
        k4_D, k4_F = rhs(D[i] + dx*k3_D, F[i] + dx*k3_F, a_e)
        
        D[i+1] = D[i] + (dx/6.0) * (k1_D + 2*k2_D + 2*k3_D + k4_D)
        F[i+1] = F[i] + (dx/6.0) * (k1_F + 2*k2_F + 2*k3_F + k4_F)
    
    # Normalize so D(a=1) = 1
    # Find index closest to a=1
    idx_1 = np.argmin(np.abs(a_grid - 1.0))
    D_norm = D / D[idx_1]
    F_norm = F / D[idx_1]
    f_grid = F_norm / np.maximum(D_norm, 1e-15)
    
    return D_norm, f_grid

def _interpolate_grid(x_query, x_grid, y_grid):
    """Linear interpolation on a uniform-ish grid."""
    if x_query <= x_grid[0]:
        return y_grid[0]
    if x_query >= x_grid[-1]:
        return y_grid[-1]
    # Binary search
    idx = np.searchsorted(x_grid, x_query)
    if idx <= 0:
        return y_grid[0]
    if idx >= len(x_grid):
        return y_grid[-1]
    t = (x_query - x_grid[idx-1]) / (x_grid[idx] - x_grid[idx-1])
    return y_grid[idx-1] + t * (y_grid[idx] - y_grid[idx-1])

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized_from_grid(a, coeffs, Om0, w0, wa, x_grid, f_grid):
    """Compute D(a)/D(1) by integrating f(exp(x)) from ln(a) to 0 using trapezoidal rule on grid."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    # Integrate f(exp(x)) from ln_a to 0
    # Use the grid f values (which are the reference f, but we need OUR f)
    # Actually, for our approximation, we need to integrate OUR f
    # This is expensive. Instead, precompute the integral for our f.
    # For speed, we'll approximate: compute our f at grid points and integrate.
    
    # For the optimization, we need fast evaluation. Let's compute our f at grid points.
    a_grid = np.exp(x_grid)
    f_ours = np.array([_f_of_a(ai, coeffs, Om0, w0, wa) for ai in a_grid])
    
    # Trapezoidal integration from ln_a to 0
    # Find indices
    idx_start = np.searchsorted(x_grid, ln_a)
    idx_end = np.searchsorted(x_grid, 0.0)
    
    if idx_start >= idx_end:
        return 1.0
    
    # Integrate from x_grid[idx_start] to x_grid[idx_end]
    x_seg = x_grid[idx_start:idx_end+1]
    f_seg = f_ours[idx_start:idx_end+1]
    
    # Add endpoints if needed
    if ln_a < x_grid[idx_start]:
        x_seg = np.concatenate([[ln_a], x_seg])
        f_seg = np.concatenate([[_f_of_a(np.exp(ln_a), coeffs, Om0, w0, wa)], f_seg])
    if 0.0 > x_grid[idx_end]:
        x_seg = np.concatenate([x_seg, [0.0]])
        f_seg = np.concatenate([f_seg, [_f_of_a(1.0, coeffs, Om0, w0, wa)]])
    
    integral_val = np.trapz(f_seg, x_seg)
    return np.exp(-integral_val)

# Precompute reference grids for a set of test cosmologies
_TEST_MODELS = [
    (-1.0, 0.0, 0.3),
    (-1.0, 0.1, 0.3),
    (-0.9, 0.0, 0.3),
    (-1.1, 0.0, 0.3),
    (-1.0, 0.3, 0.3),
    (-0.8, -0.2, 0.3),
    (-1.2, 0.5, 0.3),
    (-1.0, 0.0, 0.27),
    (-0.95, 0.1, 0.28),
    (-1.05, -0.1, 0.32),
    (-0.7, 0.0, 0.3),
    (-1.0, -0.5, 0.3),
    (-1.3, 0.0, 0.3),
    (-1.0, 0.0, 0.35),
    (-0.9, 0.2, 0.25),
]

_A_GRID = np.geomspace(0.05, 1.0, 500)
_X_GRID = np.log(_A_GRID)

def _build_reference_cache():
    """Precompute reference D and f for all test models."""
    cache = {}
    for (w0, wa, Om0) in _TEST_MODELS:
        D_ref, f_ref = _solve_growth_grid(_A_GRID, Om0, w0, wa)
        cache[(w0, wa, Om0)] = (D_ref, f_ref)
    return cache

_REF_CACHE = None

def _get_ref_cache():
    global _REF_CACHE
    if _REF_CACHE is None:
        _REF_CACHE = _build_reference_cache()
    return _REF_CACHE

def _objective_fast(coeffs):
    """Max relative error over all test models and a values, using cached reference."""
    ref_cache = _get_ref_cache()
    max_err = 0.0
    
    for (w0, wa, Om0) in _TEST_MODELS:
        D_ref_grid, f_ref_grid = ref_cache[(w0, wa, Om0)]
        
        for idx, a in enumerate(_A_GRID):
            # Our approximation
            f_approx = _f_of_a(a, coeffs, Om0, w0, wa)
            D_approx = _D_normalized_from_grid(a, coeffs, Om0, w0, wa, _X_GRID, f_ref_grid)
            
            # Reference
            f_ref = f_ref_grid[idx]
            D_ref = D_ref_grid[idx]
            
            # Relative errors
            if abs(f_ref) > 1e-10:
                err_f = abs(f_approx - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_approx - f_ref)
            
            if abs(D_ref) > 1e-10:
                err_D = abs(D_approx - D_ref) / abs(D_ref)
            else:
                err_D = abs(D_approx - D_ref)
            
            max_err = max(max_err, err_f, err_D)
    
    return max_err

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Precompute reference cache
    ref_cache = _build_reference_cache()
    _REF_CACHE = ref_cache
    
    # Check deadline
    if time.time() > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_err = _objective_fast(x0)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(
            _objective_fast,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
        )
        
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
        
        # Check deadline
        if time.time() > deadline - 10.0:
            best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
            return best_coeffs.tolist()
        
        # Try a few more local optimizations from different starting points
        starting_points = [
            [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.55, 0.05, 0.0, 0.05, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.05],
        ]
        
        for sp in starting_points:
            if time.time() > deadline - 5.0:
                break
            sp_arr = np.array(sp)
            res = minimize(
                _objective_fast,
                sp_arr,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-6}
            )
            if res.fun < best_err:
                best_coeffs = res.x
                best_err = res.fun
        
    except Exception:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
