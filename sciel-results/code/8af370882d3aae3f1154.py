"""Improved growth-index approximation via on-the-fly optimization of 6 coefficients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

# Precompute ODE reference solutions for a fixed set of cosmologies
_REFERENCE_CACHE = {}

def _get_reference(Om0, w0, wa, a_grid):
    """Solve the Riccati ODE for f(a) = d ln D / d ln a.
    
    The growth equation for k=0: D'' + (3/2)(1+w) D' = 0 where ' is d/dln(a)
    With f = D'/D: f' + f^2 + (3/2)(1+w) f = 0
    So: f' = -f^2 - (3/2)(1+w(a)) f
    """
    key = (Om0, w0, wa, tuple(a_grid))
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    
    def rhs(x, y):
        a_val = math.exp(x)
        w_val = w0 + wa * (1.0 - a_val)
        f_val = y[0]
        dfdx = -f_val * f_val - 1.5 * (1.0 + w_val) * f_val
        return [dfdx]
    
    # Integrate from early times (matter dominated, f ~ a^(3/5)) to a=1
    a_min = float(np.min(a_grid))
    x_start = math.log(max(a_min, 0.01))
    x_end = 0.0
    
    # Initial condition: f ~ a^(3/5) at early times
    f_start = max(a_min, 0.01) ** 0.6
    
    sol = solve_ivp(rhs, [x_start, x_end], [f_start],
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.05)
    
    if not sol.success:
        # Fallback: constant gamma
        gamma_ref = 0.55
        f_ref = (Om0 * a_grid**3) ** gamma_ref
        D_ref = np.exp(-np.array([_quad_f_ref(Om0, w0, wa, a, gamma_ref) for a in a_grid]))
    else:
        f_ref = np.array([sol.sol(math.log(a))[0] for a in a_grid])
        # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        D_ref = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                D_ref[i] = 1.0
            else:
                x_lower = math.log(a)
                # Integrate f from x_lower to 0
                xs = np.linspace(x_lower, 0.0, 200)
                f_vals = np.array([sol.sol(x)[0] for x in xs])
                integral = np.trapz(f_vals, xs)
                D_ref[i] = math.exp(-integral)
    
    _REFERENCE_CACHE[key] = (f_ref, D_ref)
    return f_ref, D_ref


def _quad_f_ref(Om0, w0, wa, a, gamma):
    """Quick integral for fallback."""
    def integrand(x):
        a_x = math.exp(x)
        Om_a = Om0 * a_x**3
        return Om_a ** gamma
    from scipy.integrate import quad
    if a >= 1.0:
        return 0.0
    val, _ = quad(integrand, math.log(a), 0.0, limit=50)
    return val


def _compute_gamma_vector(c, a_grid, Om0, w0, wa):
    """Compute gamma(a) for all a in a_grid."""
    Om_a = Om0 * a_grid**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a_grid)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_grid + c[5]*u*q


def _objective(c, models, a_grid, softness=20.0):
    """Compute smooth approximation to worst relative error across f and D.
    
    Uses soft-max via log-sum-exp for differentiability.
    """
    c = np.clip(c, -10.0, 10.0)
    
    all_errors = []
    
    for Om0, w0, wa in models:
        # Check gamma bounds
        gamma = _compute_gamma_vector(c, a_grid, Om0, w0, wa)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        f_approx = (Om0 * a_grid**3) ** gamma
        
        # Get reference
        f_ref, D_ref = _get_reference(Om0, w0, wa, a_grid)
        
        # Relative error for f
        denom_f = np.maximum(np.abs(f_ref), 1e-12)
        rel_err_f = np.abs(f_approx - f_ref) / denom_f
        all_errors.extend(rel_err_f.tolist())
        
        # Compute D(a)/D(1) using approximate f
        D_approx = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                D_approx[i] = 1.0
            else:
                # Integrate f_approx from ln(a) to 0
                x_lower = math.log(a)
                xs = np.linspace(x_lower, 0.0, 100)
                a_vals = np.exp(xs)
                Om_vals = Om0 * a_vals**3
                q_vals = 1.0 - Om_vals
                u_vals = 1.0 + w0 + wa * (1.0 - a_vals)
                g_vals = c[0] + c[1]*q_vals + c[2]*q_vals**2 + c[3]*u_vals + c[4]*wa*a_vals + c[5]*u_vals*q_vals
                f_vals = Om_vals ** np.clip(g_vals, 0.05, 1.25)
                integral = np.trapz(f_vals, xs)
                D_approx[i] = math.exp(-integral)
        
        denom_D = np.maximum(np.abs(D_ref), 1e-12)
        rel_err_D = np.abs(D_approx - D_ref) / denom_D
        all_errors.extend(rel_err_D.tolist())
    
    if not all_errors:
        return 1e10
    
    all_errors = np.array(all_errors)
    
    # Soft-max approximation to max
    # max(x) ≈ (1/s) * log(sum(exp(s*x)))
    scaled = softness * all_errors
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / softness
    
    return 100.0 * soft_max


def construct():
    """Return the best 6 coefficients found via optimization within time budget."""
    start_time = time.time()
    deadline = start_time + 75.0  # 15s safety margin before 90s kill
    
    # Known good constant-gamma seed
    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_coeffs = seed.copy()
    best_score = float('inf')
    
    # Test cosmologies
    TEST_COSMOLOGIES = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.3),      # CPL
        (0.25, -0.9, 0.2),     # varying
        (0.35, -1.1, 0.1),     # another
        (0.3, -1.0, -0.5),     # phantom-ish
        (0.28, -1.0, 0.0),     # Planck-like
    ]
    
    a_grid = np.linspace(0.05, 0.99, 40)
    
    # Precompute references
    for Om0, w0, wa in TEST_COSMOLOGIES:
        try:
            _get_reference(Om0, w0, wa, a_grid)
        except Exception:
            pass
    
    def obj(c):
        if time.time() > deadline:
            return 1e10
        c = np.clip(c, -10.0, 10.0)
        return _objective(c, TEST_COSMOLOGIES, a_grid, softness=30.0)
    
    # Evaluate seed
    try:
        best_score = obj(best_coeffs)
    except Exception:
        best_score = 1e10
    
    # Global bounds
    bounds = [(-10, 10)] * 6
    
    # Phase 1: Differential evolution (global search)
    try:
        if time.time() < deadline - 30:
            result_de = differential_evolution(
                obj, bounds, seed=42, maxiter=30, tol=1e-5,
                polish=True, workers=-1, init='sobol'
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_coeffs = np.clip(result_de.x, -10, 10)
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B
    try:
        if time.time() < deadline - 10:
            result_lbfgs = minimize(
                obj, best_coeffs, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = np.clip(result_lbfgs.x, -10, 10)
    except Exception:
        pass
    
    # Phase 3: Final polish with tighter tolerance
    try:
        if time.time() < deadline - 5:
            # Use a higher softness for sharper max approximation
            def obj_sharp(c):
                if time.time() > deadline:
                    return 1e10
                c = np.clip(c, -10.0, 10.0)
                return _objective(c, TEST_COSMOLOGIES, a_grid, softness=100.0)
            
            result_polish = minimize(
                obj_sharp, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result_polish.fun < best_score:
                best_score = result_polish.fun
                best_coeffs = np.clip(result_polish.x, -10, 10)
    except Exception:
        pass
    
    # Final validation: ensure coefficients are finite and in [-10, 10]
    best_coeffs = np.array(best_coeffs)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = seed
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    # Verify gamma range on a dense grid
    for Om0, w0, wa in TEST_COSMOLOGIES:
        a_dense = np.linspace(0.01, 0.999, 200)
        gamma = _compute_gamma_vector(best_coeffs, a_dense, Om0, w0, wa)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            # Fall back to seed if gamma out of range
            best_coeffs = seed
            break
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END
