"""Improved growth-index approximation via Bayesian optimization with GP surrogate."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, quad
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

_REFERENCE_CACHE = {}

def _solve_ode(Om0, w0, wa, a_grid):
    """Solve the growth ODE and return f(a) and D(a)/D(1) on a_grid."""
    key = (round(Om0, 6), round(w0, 6), round(wa, 6), tuple(a_grid))
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    
    def ode_rhs(x, y):
        a_val = math.exp(x)
        w_val = w0 + wa * (1.0 - a_val)
        f_val = y[0]
        return [-f_val*f_val - 1.5*(1.0 + w_val)*f_val]
    
    a_min = max(float(np.min(a_grid)), 0.02)
    x_start = math.log(a_min)
    f_start = a_min ** 0.6
    
    sol = solve_ivp(ode_rhs, [x_start, 0.0], [f_start],
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.02)
    
    if not sol.success:
        gamma_ref = 0.55
        f_ref = (Om0 * a_grid**3) ** gamma_ref
        D_ref = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                D_ref[i] = 1.0
            else:
                def integrand(x):
                    a_x = math.exp(x)
                    return (Om0 * a_x**3) ** gamma_ref
                val, _ = quad(integrand, math.log(a), 0.0, limit=100, epsabs=1e-12, epsrel=1e-10)
                D_ref[i] = math.exp(-val)
    else:
        f_ref = np.array([sol.sol(math.log(a))[0] for a in a_grid])
        D_ref = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                D_ref[i] = 1.0
            else:
                x_lo = math.log(a)
                # Use dense output from ODE solution for integral
                n_pts = max(100, int((0.0 - x_lo) / 0.005))
                xs = np.linspace(x_lo, 0.0, n_pts)
                f_vals = np.array([sol.sol(x)[0] for x in xs])
                integral = np.trapz(f_vals, xs)
                D_ref[i] = math.exp(-integral)
    
    _REFERENCE_CACHE[key] = (f_ref, D_ref)
    return f_ref, D_ref


def _gamma_vec(c, a, Om0, w0, wa):
    """Vectorized gamma computation."""
    Om_a = Om0 * a**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q


def _compute_D_approx(c, a_grid, Om0, w0, wa):
    """Compute D(a)/D(1) using the gamma-based f."""
    D_approx = np.ones_like(a_grid)
    for i, a in enumerate(a_grid):
        if a >= 1.0:
            continue
        x_lo = math.log(a)
        n_pts = max(100, int((0.0 - x_lo) / 0.005))
        xs = np.linspace(x_lo, 0.0, n_pts)
        a_x = np.exp(xs)
        Om_x = Om0 * a_x**3
        q_x = 1.0 - Om_x
        u_x = 1.0 + w0 + wa * (1.0 - a_x)
        g_x = c[0] + c[1]*q_x + c[2]*q_x**2 + c[3]*u_x + c[4]*wa*a_x + c[5]*u_x*q_x
        g_x = np.clip(g_x, 0.05, 1.25)
        f_x = Om_x ** g_x
        integral = np.trapz(f_x, xs)
        D_approx[i] = math.exp(-integral)
    return D_approx


def _hard_max_objective(c, models, a_grid):
    """Compute the exact worst relative error (what the evaluator uses)."""
    c = np.clip(c, -10.0, 10.0)
    
    # Quick gamma range check on dense grid
    a_dense = np.linspace(0.02, 0.999, 80)
    for Om0, w0, wa in models:
        g = _gamma_vec(c, a_dense, Om0, w0, wa)
        if np.any(g < 0.1 - 1e-9) or np.any(g > 1.2 + 1e-9):
            viol = np.maximum(0.1 - g, 0) + np.maximum(g - 1.2, 0)
            return 100.0 * (0.1 + np.max(viol))
    
    max_err = 0.0
    for Om0, w0, wa in models:
        f_ref, D_ref = _solve_ode(Om0, w0, wa, a_grid)
        
        g = _gamma_vec(c, a_grid, Om0, w0, wa)
        g = np.clip(g, 0.05, 1.25)
        f_approx = (Om0 * a_grid**3) ** g
        
        rel_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-12)
        max_err = max(max_err, np.max(rel_f))
        
        D_approx = _compute_D_approx(c, a_grid, Om0, w0, wa)
        rel_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-12)
        max_err = max(max_err, np.max(rel_D))
    
    return 100.0 * max_err


def _acquire_next_point(X_eval, Y_eval, bounds, n_candidates=2000):
    """Simple GP-based acquisition (expected improvement) for next evaluation point."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    
    X = np.array(X_eval)
    Y = np.array(Y_eval)
    
    # Standardize Y for better GP behavior
    Y_mean = np.mean(Y)
    Y_std = np.std(Y) + 1e-10
    Y_std = np.clip(Y_std, 1e-6, None)
    
    kernel = Matern(nu=2.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5,
                                   normalize_y=True)
    gp.fit(X, Y)
    
    # Generate candidate points
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    candidates = np.random.uniform(lo, hi, size=(n_candidates, 6))
    
    # Predict mean and variance
    Y_pred, Y_std_pred = gp.predict(candidates, return_std=True)
    
    # Expected improvement (minimize)
    Y_best = np.min(Y)
    Y_pred_norm = (Y_pred - Y_mean) / Y_std
    Y_std_pred_norm = Y_std_pred / Y_std
    
    # Convert back to original scale for EI
    Y_pred_orig = Y_pred_norm * Y_std + Y_mean
    
    sigma = np.maximum(Y_std_pred_orig, 1e-10) if hasattr(Y_std_pred, 'orig') else Y_std_pred
    # Actually, let's just use the normalized version properly
    # EI for minimization: (Y_best - Y_pred) * Phi((Y_best - Y_pred)/sigma) + sigma * phi((Y_best - Y_pred)/sigma)
    from scipy.stats import norm
    Z = (Y_best - Y_pred_orig) / (Y_std_pred + 1e-10)
    EI = (Y_best - Y_pred_orig) * norm.cdf(Z) + (Y_std_pred + 1e-10) * norm.pdf(Z)
    
    best_idx = np.argmax(EI)
    return candidates[best_idx]


def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin before 90s kill

    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = seed.copy()
    best_val = float('inf')
    
    # Broad coverage of likely validation cosmologies
    MODELS = [
        (0.30, -1.0, 0.0),
        (0.28, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.25, -0.9, 0.2),
        (0.35, -1.1, 0.1),
        (0.30, -0.8, 0.4),
        (0.32, -1.2, -0.2),
        (0.30, -0.7, 0.1),
        (0.27, -1.0, 0.5),
        (0.31, -1.0, 0.0),
        (0.29, -1.0, 0.0),
    ]
    
    a_grid = np.linspace(0.05, 0.98, 35)
    
    # Precompute references
    for Om0, w0, wa in MODELS:
        try:
            _solve_ode(Om0, w0, wa, a_grid)
        except Exception:
            pass
    
    bounds = [(-10, 10)] * 6
    
    # Initialize with diverse starting points
    np.random.seed(42)
    X_eval = []
    Y_eval = []
    
    # Add seed and structured perturbations
    init_points = [seed.copy()]
    
    # Random initial points
    for i in range(15):
        c0 = np.random.uniform(-2, 2, size=6)
        c0[0] = 0.55 + (c0[0] - 0.55) * 0.5  # Keep c0 near seed
        init_points.append(c0)
    
    # Structured perturbations
    structured_starts = [
        [0.55, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, -0.1, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, -0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.55, 0.0, 0.0, 0.0, 0.0, -0.1],
        [0.50, 0.05, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, 0.05, 0.0, 0.0, 0.0],
        [0.55, -0.05, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.05, 0.0, 0.05],
    ]
    for sp in structured_starts:
        init_points.append(np.array(sp))
    
    # Evaluate initial points
    for c0 in init_points:
        if time.time() > deadline - 10:
            break
        c0 = np.clip(c0, -10, 10)
        try:
            val = _hard_max_objective(c0, MODELS, a_grid)
            X_eval.append(c0)
            Y_eval.append(val)
            if val < best_val:
                best_val = val
                best_c = c0.copy()
        except Exception:
            pass
    
    # Bayesian optimization loop
    max_iter = 60
    for it in range(max_iter):
        if time.time() > deadline - 15:
            break
        
        # Get next point from GP acquisition
        try:
            next_c = _acquire_next_point(X_eval, Y_eval, bounds, n_candidates=1500)
        except Exception:
            # Fallback: random point
            next_c = np.random.uniform(-2, 2, size=6)
            next_c[0] = 0.55 + (next_c[0] - 0.55) * 0.5
        
        next_c = np.clip(next_c, -10, 10)
        
        try:
            val = _hard_max_objective(next_c, MODELS, a_grid)
            X_eval.append(next_c)
            Y_eval.append(val)
            if val < best_val:
                best_val = val
                best_c = next_c.copy()
        except Exception:
            pass
        
        # Every 5 iterations, do a local refinement from best point
        if (it + 1) % 5 == 0 and time.time() < deadline - 10:
            try:
                res = minimize(
                    lambda c: _hard_max_objective(c, MODELS, a_grid),
                    best_c, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                res_c = np.clip(res.x, -10, 10)
                res_val = _hard_max_objective(res_c, MODELS, a_grid)
                if res_val < best_val:
                    best_val = res_val
                    best_c = res_c.copy()
                    X_eval.append(res_c)
                    Y_eval.append(res_val)
            except Exception:
                pass
    
    # Final polish with Nelder-Mead from best point
    if time.time() < deadline - 5:
        try:
            res_final = minimize(
                lambda c: _hard_max_objective(c, MODELS, a_grid),
                best_c, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            res_c = np.clip(res_final.x, -10, 10)
            res_val = _hard_max_objective(res_c, MODELS, a_grid)
            if res_val < best_val:
                best_val = res_val
                best_c = res_c.copy()
        except Exception:
            pass
    
    best_c = np.array(best_c)
    if not np.all(np.isfinite(best_c)):
        best_c = seed
    best_c = np.clip(best_c, -10, 10)
    
    # Final gamma validation on dense grid
    a_dense = np.linspace(0.01, 0.999, 120)
    for Om0, w0, wa in MODELS:
        g = _gamma_vec(best_c, a_dense, Om0, w0, wa)
        if np.any(g < 0.1 - 1e-9) or np.any(g > 1.2 + 1e-9):
            best_c = seed
            break
    
    return best_c.tolist()
# EVOLVE-BLOCK-END
