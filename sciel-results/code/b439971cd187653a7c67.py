"""Refined Cohn-Goncalves Laguerre construction: continuous density collocation."""
import time
import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _build_poly_from_roots(roots, m=12):
    """Build the polynomial from Laguerre and double roots."""
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return None
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    if not np.all(np.isfinite(poly)):
        return None
    return poly


def _objective_logspace(log_roots):
    """Objective in log-space for stability."""
    roots = np.exp(log_roots)
    if np.any(roots <= 0) or np.any(roots > 1e8) or np.any(roots < 1e-8):
        return 1e10
    val = _c4_bound(roots)
    return val


def _fit_geometric(roots):
    """Fit a geometric progression to given roots. Returns (log_r0, log_q)."""
    logs = np.log(roots)
    x = np.arange(len(roots))
    A = np.vstack([np.ones_like(x), x]).T
    try:
        result = np.linalg.lstsq(A, logs, rcond=None)[0]
        return result
    except Exception:
        return (math.log(4.0), math.log(2.0))


def _geometric_roots_from_params(params):
    """Generate 12 roots from geometric parameters (log_r0, log_q)."""
    log_r0, log_q = params
    r0 = math.exp(log_r0)
    q = math.exp(log_q)
    roots = r0 * (q ** np.arange(12))
    return roots


def _objective_geometric(params):
    """Objective for geometric parameter optimization."""
    roots = _geometric_roots_from_params(params)
    if np.any(roots <= 0) or np.any(roots > 1e10) or np.any(roots < 1e-10):
        return 1e10
    return _c4_bound(roots)


def _nelder_mead(func, x0, bounds_low, bounds_high, max_iter=200):
    """Nelder-Mead minimization with bounds."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.1 * (bounds_high[i] - bounds_low[i]) if bounds_high[i] > bounds_low[i] else 1.0
        x[i] += step
        x = np.clip(x, bounds_low, bounds_high)
        simplex.append(x)
    
    func_vals = [func(x) for x in simplex]
    best_x = x0[:]
    best_val = func(best_x)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        order = np.argsort(func_vals)
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        if func_vals[0] < best_val:
            best_val = func_vals[0]
            best_x = simplex[0].copy()
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds_low, bounds_high)
        fr = func(xr)
        
        if fr <= func_vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds_low, bounds_high)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                func_vals[-1] = fe
            else:
                simplex[-1] = xr
                func_vals[-1] = fr
        elif fr < func_vals[-2]:
            simplex[-1] = xr
            func_vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds_low, bounds_high)
            fc = func(xc)
            if fc < func_vals[-1]:
                simplex[-1] = xc
                func_vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds_low, bounds_high)
                    func_vals[i] = func(simplex[i])
    
    return best_x, best_val


def _density_collocation():
    """
    Model root positions as stationary points of a 1D Fokker-Planck equation.
    The density profile rho(x) satisfies a nonlinear boundary value problem.
    We use finite-difference collocation on a log-spaced grid.
    
    The potential is V(x) = a*x + b*log(x) + c/x (harmonic + log repulsion + inverse)
    Stationary condition: d/dx [rho * (V' + rho')] = 0
    """
    if _time_left() < 60:
        return None
    
    # Work in log-space: let u = log(x), so x = exp(u)
    # The density in u-space: sigma(u) = rho(exp(u)) * exp(u)
    # We want 12 roots, so integrate sigma over u to get 12
    
    # Grid in u-space
    n_grid = 200
    u_min = math.log(0.5)
    u_max = math.log(200.0)
    u_grid = np.linspace(u_min, u_max, n_grid)
    du = u_grid[1] - u_grid[0]
    
    # Potential parameters to optimize: a, b, c
    # V(x) = a*x + b*log(x) + c/x
    # In u-space: V(exp(u)) = a*exp(u) + b*u + c*exp(-u)
    # V'(x) = a + b/x - c/x^2
    # V'(exp(u)) = a + b*exp(-u) - c*exp(-2u)
    
    def _stationary_density(params, u_grid):
        """Compute stationary density sigma(u) for given potential params."""
        a, b, c = params
        # V'(exp(u)) = a + b*exp(-u) - c*exp(-2u)
        Vprime = a + b * np.exp(-u_grid) - c * np.exp(-2 * u_grid)
        
        # Stationary Fokker-Planck: sigma(u) = C * exp(-V(exp(u)) + integral of something)
        # Actually, for 1D Fokker-Planck with potential V:
        # sigma(u) ∝ exp(-V(exp(u))) * exp(u)  (Jacobian from x to u)
        # But with the nonlinear term, it's more complex.
        # Simplification: use the linearized stationary solution
        # sigma(u) ∝ exp(-V(exp(u))) * exp(u)
        
        V_u = a * np.exp(u_grid) + b * u_grid + c * np.exp(-u_grid)
        sigma = np.exp(-V_u + u_grid)
        
        # Normalize to integrate to 12
        integral = np.trapz(sigma, u_grid)
        if integral > 0 and np.isfinite(integral):
            sigma = sigma * 12.0 / integral
        else:
            sigma = np.zeros_like(u_grid)
        
        return sigma
    
    def _roots_from_density(sigma, u_grid):
        """Extract 12 root positions from density profile using quantiles."""
        # Find the 12 quantiles of the density
        cumulative = np.cumsum(sigma)
        total = cumulative[-1]
        if total <= 0:
            return None
        
        # Target quantiles: 1/12, 2/12, ..., 12/12
        targets = np.arange(1, 13) / 12.0 * total
        
        # Interpolate to find u values
        u_roots = []
        for target in targets:
            idx = np.searchsorted(cumulative, target)
            if idx == 0:
                u_roots.append(u_grid[0])
            elif idx >= len(u_grid):
                u_roots.append(u_grid[-1])
            else:
                # Linear interpolation
                frac = (target - cumulative[idx-1]) / (cumulative[idx] - cumulative[idx-1])
                u_roots.append(u_grid[idx-1] + frac * (u_grid[idx] - u_grid[idx-1]))
        
        roots = np.exp(u_roots)
        return roots
    
    def _objective_density(params):
        """Objective for density collocation."""
        sigma = _stationary_density(params, u_grid)
        roots = _roots_from_density(sigma, u_grid)
        if roots is None or np.any(roots <= 0):
            return 1e10
        return _c4_bound(roots)
    
    # Optimize potential parameters
    initial_params = [(1.0, 2.0, 1.0), (0.5, 3.0, 0.5), (2.0, 1.0, 2.0), (1.0, 1.0, 0.5)]
    
    best_val = float('inf')
    best_roots = None
    
    for init in initial_params:
        if _time_left() < 30:
            break
        try:
            result = minimize(
                _objective_density,
                x0=np.array(init),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                sigma = _stationary_density(result.x, u_grid)
                best_roots = _roots_from_density(sigma, u_grid)
        except Exception:
            continue
    
    if best_roots is not None and best_val < float('inf'):
        return [float(r) for r in best_roots], best_val
    return None


def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 0: Try density collocation (new structural approach)
    if _time_left() > 180:
        result = _density_collocation()
        if result is not None:
            roots_candidate, val_candidate = result
            if val_candidate < best_val:
                best_val = val_candidate
                best_roots = roots_candidate
    
    # Phase 1: Optimize in geometric parameter space
    if _time_left() > 120:
        np.random.seed(42)
        
        # Fit geometric to seed
        seed_logs = np.log(seed_roots)
        x = np.arange(n_roots)
        A = np.vstack([np.ones_like(x), x]).T
        try:
            log_r0_fit, log_q_fit = np.linalg.lstsq(A, seed_logs, rcond=None)[0]
        except Exception:
            log_r0_fit, log_q_fit = math.log(4.0), math.log(2.0)
        
        start_points = [
            (log_r0_fit, log_q_fit),
            (math.log(4.0), math.log(2.0)),
            (math.log(2.0), math.log(2.0)),
            (math.log(1.0), math.log(1.5)),
            (math.log(5.0), math.log(1.3)),
            (math.log(3.0), math.log(1.8)),
            (math.log(1.0), math.log(2.0)),
            (math.log(10.0), math.log(1.2)),
            (math.log(0.5), math.log(2.5)),
        ]
        
        for _ in range(8):
            lr0 = np.random.uniform(math.log(0.1), math.log(20.0))
            lq = np.random.uniform(math.log(0.5), math.log(4.0))
            start_points.append((lr0, lq))
        
        for sp in start_points:
            if _time_left() < 60:
                break
            
            try:
                result = minimize(
                    _objective_geometric,
                    x0=np.array(sp),
                    method='L-BFGS-B',
                    bounds=[(math.log(0.01), math.log(100.0)), (math.log(0.1), math.log(10.0))],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    roots = _geometric_roots_from_params(result.x)
                    if np.all(roots > 0) and np.all(np.isfinite(roots)):
                        score = _c4_bound(roots, m)
                        if score < best_val:
                            best_val = score
                            best_roots = [float(r) for r in roots]
            except Exception:
                continue
    
    # Phase 2: Direct optimization of all 12 roots in log-space
    if _time_left() > 60:
        def objective_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0):
                return float('inf')
            return _c4_bound(roots, m)
        
        x0 = np.log(np.array(best_roots, dtype=float))
        bounds_low = np.ones(n_roots) * math.log(0.05)
        bounds_high = np.ones(n_roots) * math.log(200.0)
        
        try:
            result = minimize(
                objective_log,
                x0=x0,
                method='L-BFGS-B',
                bounds=list(zip(bounds_low, bounds_high)),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if np.isfinite(result.fun) and result.fun < best_val:
                roots = np.exp(result.x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = result.fun
                    best_roots = [float(r) for r in roots]
        except Exception:
            pass
        
        # Nelder-Mead in log-space
        if _time_left() > 30:
            x0 = np.log(np.array(best_roots, dtype=float))
            best_x, best_f = _nelder_mead(
                objective_log, x0, bounds_low, bounds_high, max_iter=200
            )
            
            if np.isfinite(best_f) and best_f < best_val:
                roots = np.exp(best_x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = best_f
                    best_roots = [float(r) for r in roots]
    
    # Phase 3: Fine-tune with perturbations
    if _time_left() > 30 and best_val < seed_val:
        def objective2_log(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0):
                return float('inf')
            return _c4_bound(roots, m)
        
        x0 = np.log(np.array(best_roots, dtype=float))
        bounds_low = np.ones(n_roots) * math.log(0.02)
        bounds_high = np.ones(n_roots) * math.log(300.0)
        
        for restart in range(5):
            if _time_left() < 20:
                break
            x0_pert = x0 + np.random.normal(0, 0.1, n_roots)
            x0_pert = np.clip(x0_pert, bounds_low, bounds_high)
            
            best_x, best_f = _nelder_mead(
                objective2_log, x0_pert, bounds_low, bounds_high, max_iter=100
            )
            
            if np.isfinite(best_f) and best_f < best_val:
                roots = np.exp(best_x)
                if np.all(roots > 0) and np.all(np.isfinite(roots)):
                    best_val = best_f
                    best_roots = [float(r) for r in roots]
                    x0 = best_x
    
    return best_roots
