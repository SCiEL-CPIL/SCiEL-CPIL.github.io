"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

_DEADLINE_SECONDS = 90.0
_DEADLINE_FRACTION = 0.85

def _compute_gamma(c, a, Om, w0, wa):
    """Compute gamma(a) for given coefficients and scale factor."""
    Om_a = Om * a**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_from_gamma(c, a, Om, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    g = _compute_gamma(c, a, Om, w0, wa)
    Om_a = Om * a**3
    return np.power(Om_a, g)

def _normalized_growth(c, a_grid, Om, w0, wa, n_int=1000):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    ln_a = np.log(a_grid)
    D_ratio = np.ones_like(a_grid)
    for i, la in enumerate(ln_a):
        xs = np.linspace(la, 0.0, n_int)
        a_vals = np.exp(xs)
        f_vals = _f_from_gamma(c, a_vals, Om, w0, wa)
        integral = np.trapz(f_vals, xs)
        D_ratio[i] = np.exp(-integral)
    return D_ratio

def _validate_gamma(c, a_grid, Om, w0, wa):
    """Check that gamma is in [0.1, 1.2] for all points."""
    g = _compute_gamma(c, a_grid, Om, w0, wa)
    return np.all(g >= 0.1 - 1e-9) and np.all(g <= 1.2 + 1e-9)

def _solve_ode_f(a_grid, w0, wa):
    """Solve the ODE for f(a) = dlnD/dlna numerically."""
    def rhs(lna, y):
        a_val = np.exp(lna)
        w_val = w0 + wa * (1.0 - a_val)
        f_val = y[0]
        return [-(1.0/2.0)*f_val**2 - 1.5*(1.0+w_val)*f_val]
    
    a_min = np.min(a_grid)
    ln_a_min = np.log(a_min)
    
    sol = solve_ivp(rhs, [0.0, ln_a_min], [0.4], method='RK45', 
                    rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.01)
    
    if not sol.success:
        return None
    
    f_ref = np.array([sol.sol(la)[0] for la in np.log(a_grid)])
    return f_ref, sol

def _normalized_growth_ref(a_grid, sol):
    """Compute normalized growth factor from ODE solution."""
    D_ratio = np.ones_like(a_grid)
    for i, la in enumerate(np.log(a_grid)):
        xs = np.linspace(la, 0.0, 500)
        f_vals = np.array([sol.sol(x)[0] for x in xs])
        integral = np.trapz(f_vals, xs)
        D_ratio[i] = np.exp(-integral)
    return D_ratio

def _objective(c, Om, w0, wa, a_grid, f_ref, D_ref):
    """Compute the worst relative error for f and D."""
    if not _validate_gamma(c, a_grid, Om, w0, wa):
        return 1e10
    
    f_approx = _f_from_gamma(c, a_grid, Om, w0, wa)
    
    denom_f = np.maximum(np.abs(f_ref), 1e-12)
    rel_err_f = np.abs(f_approx - f_ref) / denom_f
    
    D_approx_ratio = _normalized_growth(c, a_grid, Om, w0, wa, n_int=500)
    
    denom_D = np.maximum(np.abs(D_ref), 1e-12)
    rel_err_D = np.abs(D_approx_ratio - D_ref) / denom_D
    
    worst_err = max(np.max(rel_err_f), np.max(rel_err_D))
    return 100.0 * worst_err

def _lsq_prefit(models, a_grid):
    """
    Perform a least-squares pre-fit in log-space to get a good initial point.
    ln(f) = gamma * ln(Om(a)) is linear in the coefficients c.
    We fit gamma to minimize ||ln(f_approx) - ln(f_ref)||^2.
    """
    # Build the design matrix and target
    A_rows = []
    b_rows = []
    
    for Om, w0, wa in models:
        f_ref, sol = _solve_ode_f(a_grid, w0, wa)
        if f_ref is None:
            continue
        
        # Only use points where f_ref > 0
        valid = f_ref > 1e-10
        if np.sum(valid) < 10:
            continue
        
        a_valid = a_grid[valid]
        f_ref_valid = f_ref[valid]
        
        for a_val in a_valid:
            Om_a = Om * a_val**3
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_val)
            
            # Design row: [1, q, q^2, u, wa*a, u*q]
            row = np.array([1.0, q, q**2, u, wa*a_val, u*q])
            A_rows.append(row)
            
            # Target: ln(f_ref) / ln(Om_a) = gamma
            # ln(f) = gamma * ln(Om_a) => gamma = ln(f) / ln(Om_a)
            if Om_a > 1e-10 and Om_a < 1.0:
                gamma_target = np.log(f_ref_valid[np.where(a_grid == a_val)[0][0]]) / np.log(Om_a)
            else:
                gamma_target = 0.55  # fallback
            b_rows.append(gamma_target)
    
    if len(A_rows) == 0:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Solve least squares
    try:
        c_fit, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        c_fit = np.clip(c_fit, -10.0, 10.0)
        
        # Validate gamma bounds
        all_valid = True
        for Om, w0, wa in models:
            if not _validate_gamma(c_fit, a_grid, Om, w0, wa):
                all_valid = False
                break
        
        if all_valid:
            return c_fit
    except Exception:
        pass
    
    return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

def _multi_model_objective(c, models, a_grid, refs):
    """Evaluate worst error across all models."""
    max_err = 0.0
    for i, (Om, w0, wa) in enumerate(models):
        f_ref, D_ref = refs[i]
        if f_ref is None:
            continue
        err = _objective(c, Om, w0, wa, a_grid, f_ref, D_ref)
        if err > max_err:
            max_err = err
        if err >= 1e10:
            return 1e10
    return max_err

# EVOLVE-BLOCK-END

# Global cache for models
_models = None
_a_grid = None
_refs_cache = None

def _get_models():
    global _models
    if _models is None:
        _models = [
            (0.3, -0.5, 0.5),
            (0.3, -1.0, 0.0),
            (0.3, -0.9, 0.3),
            (0.25, -0.8, 0.2),
            (0.35, -1.0, 0.5),
            (0.3, -0.7, -0.2),
            (0.28, -1.0, 0.8),
            (0.32, -0.6, 0.1),
        ]
    return _models

def _get_a_grid():
    global _a_grid
    if _a_grid is None:
        _a_grid = np.logspace(-1, 0, 50)
    return _a_grid

def _get_refs(models, a_grid):
    """Pre-compute ODE references for all models."""
    global _refs_cache
    if _refs_cache is None:
        _refs_cache = []
        for Om, w0, wa in models:
            f_ref, sol = _solve_ode_f(a_grid, w0, wa)
            if f_ref is None:
                _refs_cache.append((None, None))
            else:
                D_ref = _normalized_growth_ref(a_grid, sol)
                _refs_cache.append((f_ref, D_ref))
    return _refs_cache

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS * _DEADLINE_FRACTION
    
    models = _get_models()
    a_grid = _get_a_grid()
    refs = _get_refs(models, a_grid)
    
    # Step 1: LSQ pre-fit in log-space
    best_c = _lsq_prefit(models, a_grid)
    best_obj = _multi_model_objective(best_c, models, a_grid, refs)
    
    # Define the objective function for the optimizer
    def obj(c):
        if time.time() > deadline:
            return 1e10
        c = np.clip(c, -10.0, 10.0)
        return _multi_model_objective(c, models, a_grid, refs)
    
    # Step 2: Minimax polish with Nelder-Mead from the pre-fit point
    np.random.seed(42)
    
    # Try from the pre-fit point
    if time.time() < deadline - 15:
        try:
            res = minimize(obj, best_c, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_c = np.clip(res.x, -10.0, 10.0)
                best_obj = res.fun
        except Exception:
            pass
    
    # Try from the seed
    c_seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    if time.time() < deadline - 10:
        try:
            res2 = minimize(obj, c_seed, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            if res2.fun < best_obj:
                best_c = np.clip(res2.x, -10.0, 10.0)
                best_obj = res2.fun
        except Exception:
            pass
    
    # Step 3: Local refinement around best
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            try:
                res3 = minimize(obj, best_c, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
                if res3.fun < best_obj:
                    best_c = np.clip(res3.x, -10.0, 10.0)
                    best_obj = res3.fun
            except Exception:
                pass
    
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]

# EVOLVE-BLOCK-END
