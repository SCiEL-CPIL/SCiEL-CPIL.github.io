"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 80.0  # Safety margin before 90s kill

_REF_DATA = None

def _om_a_vec(a, Om0, w0, wa):
    """Vectorized matter density fraction."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.maximum(a, 1e-10)
    w_eff = 1.0 + w0 + wa
    de_term = (1.0 - Om0) * np.exp(-3.0 * w_eff * np.log(a_safe) + 3.0 * wa * a_safe)
    m_term = Om0 * np.exp(-3.0 * np.log(a_safe))
    E2 = m_term + de_term
    return m_term / E2

def _gamma_vec(a, Om0, w0, wa, c):
    """Vectorized gamma."""
    c0, c1, c2, c3, c4, c5 = c
    Om_a = _om_a_vec(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_vec(a, Om0, w0, wa, c):
    """Vectorized f = Om(a)^gamma."""
    Om_a = np.maximum(_om_a_vec(a, Om0, w0, wa), 1e-12)
    g = _gamma_vec(a, Om0, w0, wa, c)
    return np.exp(g * np.log(Om_a))

def _D_vec(a, Om0, w0, wa, c):
    """Vectorized normalized D(a)/D(1)."""
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    a_m = a[mask]
    a_min = max(np.min(a_m), 1e-6)
    n = 800
    ln_grid = np.linspace(np.log(a_min), 0.0, n)
    a_grid = np.exp(ln_grid)
    f_grid = _f_vec(a_grid, Om0, w0, wa, c)
    dx = np.diff(ln_grid)
    trap = 0.5 * (f_grid[1:] + f_grid[:-1]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    total = cum[-1]
    ln_m = np.log(a_m)
    I = np.interp(ln_m, ln_grid, cum) - total
    D[mask] = np.exp(I)
    return D

def _build_refs():
    """Build reference data using ODE."""
    global _REF_DATA
    if _REF_DATA is not None:
        return _REF_DATA
    
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.30, -1.5, 1.0),
        (0.30, -0.5, -0.5),
        (0.20, -1.2, 0.8),
        (0.40, -0.7, -0.3),
        (0.28, -1.0, 0.1),
        (0.32, -0.95, -0.15),
        (0.30, -1.0, 0.3),
        (0.30, -0.85, 0.25),
        (0.27, -1.1, 0.4),
        (0.33, -0.75, -0.25),
    ]
    
    # Use a moderately dense grid for evaluation to balance accuracy and speed
    a_vals = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ln_a_eval = np.log(a_vals)
    
    ref_models = []
    for Om0, w0, wa in models:
        def ode(ln_a, y):
            f = y[0]
            a = np.exp(ln_a)
            Om_a = _om_a_vec(np.array([a]), Om0, w0, wa)[0]
            w_eff = 1.0 + w0 + wa
            g = a**(-3.0 * w_eff) * np.exp(3.0 * wa * a)
            E2 = Om0 * a**(-3.0) + (1.0 - Om0) * g
            dE2_da = -3.0 * Om0 * a**(-4.0) + (1.0 - Om0) * g * 3.0 * (wa - w_eff / a)
            dlnE2_dlna = dE2_da * a / E2
            HpH = 0.5 * dlnE2_dlna
            fp = -1.5 * Om_a / E2 - (3.0 / a + HpH) * f - f * f
            return [fp]
        
        f0 = 0.55 * Om0**0.55
        try:
            sol = solve_ivp(ode, [0.0, np.log(0.15)], [f0], t_eval=ln_a_eval,
                            method='RK45', rtol=1e-10, atol=1e-12)
            f_ref = sol.y[0] if sol.success else None
        except Exception:
            f_ref = None
            
        if f_ref is None or len(f_ref) != len(a_vals):
            # Fallback to seed approximation if ODE fails
            f_ref = _f_vec(a_vals, Om0, w0, wa, [0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        # Reference D via trapezoid on dense grid
        n_d = 2000
        ln_dense = np.linspace(np.log(0.15), 0.0, n_d)
        a_dense = np.exp(ln_dense)
        f_dense = np.interp(ln_dense, ln_a_eval, f_ref)
        dx = np.diff(ln_dense)
        trap = 0.5 * (f_dense[1:] + f_dense[:-1]) * dx
        cum = np.concatenate([[0.0], np.cumsum(trap)])
        total = cum[-1]
        I = np.interp(ln_a_eval, ln_dense, cum) - total
        D_ref = np.exp(I)
        
        ref_models.append({
            'Om0': Om0, 'w0': w0, 'wa': wa,
            'a': a_vals, 'f_ref': f_ref, 'D_ref': D_ref
        })
        
    _REF_DATA = ref_models
    return ref_models

def _score(c):
    """Worst relative error (0-1 scale) plus gamma-constraint penalty."""
    if _REF_DATA is None:
        _build_refs()
    refs = _REF_DATA
    max_err = 0.0
    gamma_violation = 0.0
    
    for m in refs:
        Om0, w0, wa = m['Om0'], m['w0'], m['wa']
        a = m['a']
        f_ref = m['f_ref']
        D_ref = m['D_ref']
        
        f_ap = _f_vec(a, Om0, w0, wa, c)
        D_ap = _D_vec(a, Om0, w0, wa, c)
        
        ef = np.abs(f_ap - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        eD = np.abs(D_ap - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        max_err = max(max_err, float(np.max(ef)), float(np.max(eD)))
        
        g = _gamma_vec(a, Om0, w0, wa, c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_violation = max(gamma_violation, float(np.max(np.abs(np.clip(g, 0.1, 1.2) - g))))
            
    if gamma_violation > 0:
        return max_err + 10.0 * gamma_violation
        
    return max_err

def _score_smooth(c, beta=20.0):
    """Smooth surrogate using log-sum-exp for the max operation."""
    if _REF_DATA is None:
        _build_refs()
    refs = _REF_DATA
    all_errors = []
    gamma_violation = 0.0
    
    for m in refs:
        Om0, w0, wa = m['Om0'], m['w0'], m['wa']
        a = m['a']
        f_ref = m['f_ref']
        D_ref = m['D_ref']
        
        f_ap = _f_vec(a, Om0, w0, wa, c)
        D_ap = _D_vec(a, Om0, w0, wa, c)
        
        ef = np.abs(f_ap - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        eD = np.abs(D_ap - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        all_errors.extend(ef.tolist())
        all_errors.extend(eD.tolist())
        
        g = _gamma_vec(a, Om0, w0, wa, c)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_violation = max(gamma_violation, float(np.max(np.abs(np.clip(g, 0.1, 1.2) - g))))
            
    arr = np.array(all_errors)
    max_val = float(np.max(arr))
    
    if gamma_violation > 0:
        penalty = 10.0 * gamma_violation
        return float(np.logaddexp(np.log(max_val + 1e-15), np.log(penalty + 1e-15)))
    
    if max_val < 1e-15:
        return max_val
    smooth_max = max_val + np.log(np.sum(np.exp(arr - max_val))) / beta
    return float(smooth_max)

def _score_grad(c):
    """Score with numerical gradient for L-BFGS-B."""
    val = _score(c)
    eps = 1e-7
    grad = np.zeros(6)
    for i in range(6):
        c_plus = c.copy()
        c_plus[i] += eps
        c_minus = c.copy()
        c_minus[i] -= eps
        grad[i] = (_score(c_plus) - _score(c_minus)) / (2 * eps)
    return val, grad

def _optimize():
    """Run global and local optimization."""
    # Physics-based initial guess
    seed = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_c = seed.copy()
    best_score = _score(seed)
    
    bounds = [(-10.0, 10.0)] * 6
    
    # 1. Differential Evolution for global exploration (use smooth surrogate for speed)
    if time.time() < _DEADLINE - 25.0:
        try:
            de_res = differential_evolution(
                _score_smooth, bounds, 
                maxiter=150, popsize=25, 
                tol=1e-6, 
                seed=42, 
                polish=False
            )
            # Validate with exact score
            de_score = _score(de_res.x)
            if de_score < best_score:
                best_c = de_res.x
                best_score = de_score
        except Exception:
            pass

    # 2. Multi-start Nelder-Mead on smooth surrogate, then exact polish
    if time.time() < _DEADLINE - 12.0:
        starts = [
            best_c,
            np.array([0.5, 0.1, -0.05, 0.1, 0.0, 0.0]),
            np.array([0.6, -0.1, 0.1, 0.0, 0.0, 0.05]),
            np.array([0.55, 0.05, 0.0, 0.05, -0.05, 0.0]),
            np.array([0.5, 0.0, 0.1, -0.1, 0.05, -0.05]),
            np.array([0.55, -0.05, 0.05, 0.0, 0.0, 0.05]),
            np.array([0.6, 0.0, 0.0, 0.1, 0.0, 0.0]),
            np.array([0.5, 0.1, 0.0, 0.0, 0.05, 0.05]),
            np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.6, 0.05, -0.05, 0.05, 0.05, -0.03]),
            np.array([0.5, -0.05, 0.08, -0.08, 0.02, 0.04]),
            np.array([0.58, 0.02, 0.03, -0.02, -0.03, 0.02]),
        ]
        
        for start in starts:
            if time.time() > _DEADLINE - 12.0:
                break
            try:
                # Smooth surrogate for fast convergence
                nm_res = minimize(
                    _score_smooth, start, 
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True}
                )
                nm_c = np.clip(nm_res.x, -10, 10)
                nm_score = _score(nm_c)
                if nm_score < best_score:
                    best_c = nm_c
                    best_score = nm_score
            except Exception:
                pass

    # 3. Random restarts with exact score, then polish
    if time.time() < _DEADLINE - 8.0:
        rng = np.random.default_rng(12345)
        base_scale = [0.1, 0.3, 0.1, 0.3, 0.3, 0.2]
        for restart in range(300):
            if time.time() > _DEADLINE - 8.0:
                break
            scale_factor = max(0.01, 1.0 - restart / 300.0)
            scale = [s * scale_factor for s in base_scale]
            c = rng.normal(loc=best_c, scale=scale)
            c = np.clip(c, -10, 10)
            s = _score(c)
            if s < best_score:
                best_score = s
                best_c = c.copy()
                # Polish with smooth surrogate then exact
                if time.time() < _DEADLINE - 8.0:
                    try:
                        res_smooth = minimize(
                            _score_smooth, best_c, 
                            method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True}
                        )
                        smooth_c = np.clip(res_smooth.x, -10, 10)
                        smooth_score = _score(smooth_c)
                        if smooth_score < best_score:
                            best_c = smooth_c
                            best_score = smooth_score
                    except Exception:
                        pass
                    if time.time() < _DEADLINE - 8.0:
                        try:
                            res = minimize(
                                _score, best_c, 
                                method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-11, 'adaptive': True}
                            )
                            if res.fun < best_score:
                                best_score = res.fun
                                best_c = np.clip(res.x, -10, 10)
                        except Exception:
                            pass

    # 4. Final high-precision polish
    if time.time() < _DEADLINE - 2.0:
        # L-BFGS-B with numerical gradient
        try:
            lbfgs_res = minimize(
                _score, best_c, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if lbfgs_res.fun < best_score:
                best_c = lbfgs_res.x
                best_score = lbfgs_res.fun
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() < _DEADLINE - 2.0:
            try:
                nm_res2 = minimize(
                    _score, best_c, 
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-12, 'adaptive': True}
                )
                if nm_res2.fun < best_score:
                    best_c = nm_res2.x
                    best_score = nm_res2.fun
            except Exception:
                pass
        
        # Fine random search
        if time.time() < _DEADLINE - 1.0:
            rng_final = np.random.default_rng(999)
            for _ in range(100):
                if time.time() > _DEADLINE - 1.0:
                    break
                c = rng_final.normal(loc=best_c, scale=[0.003, 0.008, 0.003, 0.008, 0.008, 0.006])
                c = np.clip(c, -10, 10)
                s = _score(c)
                if s < best_score:
                    best_score = s
                    best_c = c.copy()

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c

# Precompute references at import
try:
    _build_refs()
except Exception:
    _REF_DATA = None

_optimized = None
def construct():
    global _optimized
    if _optimized is None:
        _optimized = _optimize()
    return list(_optimized)
