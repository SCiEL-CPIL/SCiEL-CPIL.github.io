# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.special import factorial

_T0 = time.time()
_DEADLINE = 195.0  # Safety margin below the 210s kill

# ---------- Precompute Hermite Polynomial Coefficients (hardcoded, low->high) ----------
_H0_coeffs = np.zeros(13)
_H0_coeffs[0] = 1.0

_H4_coeffs = np.zeros(13)
_H4_coeffs[0] = 12.0
_H4_coeffs[2] = -48.0
_H4_coeffs[4] = 16.0

_H8_coeffs = np.zeros(13)
_H8_coeffs[0] = 40320.0
_H8_coeffs[2] = -67200.0
_H8_coeffs[4] = 26880.0
_H8_coeffs[6] = -4480.0
_H8_coeffs[8] = 256.0

_H12_coeffs = np.zeros(13)
_H12_coeffs[0] = 1032160.0
_H12_coeffs[2] = -393216000.0
_H12_coeffs[4] = 145152000.0
_H12_coeffs[6] = -29030400.0
_H12_coeffs[8] = 2903040.0
_H12_coeffs[10] = -147456.0
_H12_coeffs[12] = 4096.0

_H0_0 = _H0_coeffs[0]
_H4_0 = _H4_coeffs[0]
_H8_0 = _H8_coeffs[0]
_H12_0 = _H12_coeffs[0]

def _q_coeffs(c0, c1, c2):
    """
    Compute the coefficients of q(x) = g(x)/x^2.
    g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    """
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # g(x) coefficients (degree 0 to 12)
    g = c0 * _H0_coeffs + c1 * _H4_coeffs + c2 * _H8_coeffs + c3 * _H12_coeffs
    
    # q(x) = g(x) / x^2. Since g is even, g[1] is 0.
    # q coefficients are g[2], g[3], ..., g[12] shifted to degree 0 to 10.
    q_full = g[2:] 
    return q_full

def _find_sign_changing_roots(q_coeffs, tol=1e-10):
    """
    Find all positive real roots of q where sign changes (simple roots).
    Returns sorted list of such roots.
    """
    qc = q_coeffs.copy()
    while len(qc) > 1 and abs(qc[-1]) < 1e-14:
        qc = qc[:-1]
    
    if len(qc) < 2:
        return []
        
    try:
        roots = np.roots(qc)
    except Exception:
        return []
        
    # Compute derivative for sign-change detection
    d_coeffs = np.polyder(np.flip(qc))
    
    valid_roots = []
    for rt in roots:
        if abs(rt.imag) > tol * max(1.0, abs(rt.real)):
            continue
        zr = float(rt.real)
        if zr <= 1e-14:
            continue
            
        # Check residual
        q_val = np.polyval(np.flip(qc), zr)
        if abs(q_val) > 1e-8 * max(1.0, abs(zr)):
            continue
            
        # Check sign change: derivative should be non-zero
        d_val = np.polyval(d_coeffs, zr)
        if abs(d_val) < 1e-10 * max(1.0, abs(zr)):
            continue
            
        valid_roots.append(zr)
    
    return sorted(valid_roots)

def _objective(params):
    """
    Objective function for optimization.
    Returns value = r^2 / (2*pi). Lower is better.
    """
    c0, c1, c2 = params
    try:
        qc = _q_coeffs(c0, c1, c2)
        roots = _find_sign_changing_roots(qc)
        if not roots:
            return 1e12
        r = roots[0]  # smallest positive sign-changing root
        return r * r / (2.0 * np.pi)
    except Exception:
        return 1e12

def _smooth_objective(params, soft_min_temp=0.01):
    """
    Smooth surrogate: use soft-min over the smallest few roots to create
    a differentiable objective that guides optimization toward smaller roots.
    """
    c0, c1, c2 = params
    try:
        qc = _q_coeffs(c0, c1, c2)
        roots = _find_sign_changing_roots(qc)
        if not roots:
            return 1e12
        
        # Use soft-min over the smallest 3 roots (if they exist)
        k = min(3, len(roots))
        r_vals = np.array(roots[:k])
        
        # Soft-min: -T * log(sum(exp(-r_i/T))) approximates min(r_i) for small T
        scaled = -r_vals / soft_min_temp
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -soft_min_temp * log_sum
        
        return soft_min_val * soft_min_val / (2.0 * np.pi)
    except Exception:
        return 1e12

def _seed():
    return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

def _multi_start_local(x0, max_time_budget=30.0):
    """
    Perform local refinement from multiple nearby starting points.
    """
    best_val = _objective(x0)
    best_x = x0.copy()
    
    start_time = time.time()
    
    # Try perturbations in coefficient space
    np.random.seed(42)
    for trial in range(20):
        if time.time() - start_time > max_time_budget:
            break
        
        # Random perturbation with decreasing scale
        scale = 0.1 / (1.0 + trial * 0.1)
        perturbation = np.random.normal(0, scale, size=3)
        x_try = best_x + perturbation
        
        val = _objective(x_try)
        if val < best_val:
            best_val = val
            best_x = x_try.copy()
    
    # Nelder-Mead refinement
    if time.time() - start_time < max_time_budget - 5:
        try:
            res = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Smooth objective refinement to escape local minima
    if time.time() - start_time < max_time_budget - 3:
        try:
            res_smooth = minimize(
                _smooth_objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True}
            )
            if np.isfinite(res_smooth.fun):
                val = _objective(res_smooth.x)
                if val < best_val:
                    best_val = val
                    best_x = res_smooth.x.copy()
        except Exception:
            pass
    
    return best_x, best_val

def _high_precision_polish(x0, max_time_budget=10.0):
    """
    Final high-precision polish using very tight tolerances.
    """
    best_val = _objective(x0)
    best_x = x0.copy()
    
    start_time = time.time()
    
    # Very tight Nelder-Mead
    if time.time() - start_time < max_time_budget:
        try:
            res = minimize(
                _objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-16, 'fatol': 1e-17, 'adaptive': True}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Coordinate descent refinement
    if time.time() - start_time < max_time_budget - 2:
        for coord in range(3):
            if time.time() - start_time > max_time_budget - 1:
                break
            try:
                def _1d_obj(xi):
                    x_try = best_x.copy()
                    x_try[coord] = xi
                    return _objective(x_try)
                
                res_1d = minimize_scalar(
                    _1d_obj,
                    bounds=(best_x[coord] - 1e-6, best_x[coord] + 1e-6),
                    method='bounded',
                    options={'xatol': 1e-15, 'maxiter': 200}
                )
                if np.isfinite(res_1d.fun) and res_1d.fun < best_val:
                    best_x[coord] = res_1d.x
                    best_val = res_1d.fun
            except Exception:
                pass
    
    return best_x, best_val

def construct():
    global _T0
    try:
        seed = _seed()
        best_val = _objective(seed)
        best_x = seed.copy()
        
        # Phase 1: Differential Evolution for global exploration
        if time.time() - _T0 < _DEADLINE - 60:
            try:
                bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
                res = differential_evolution(
                    _objective, 
                    bounds, 
                    seed=42, 
                    maxiter=150, 
                    popsize=30, 
                    tol=1e-12,
                    polish=False
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 2: Multi-start local refinement from best candidate
        if time.time() - _T0 < _DEADLINE - 25:
            try:
                x_local, val_local = _multi_start_local(best_x, max_time_budget=35.0)
                if val_local < best_val:
                    best_val = val_local
                    best_x = x_local.copy()
            except Exception:
                pass
        
        # Phase 3: High-precision polish
        if time.time() - _T0 < _DEADLINE - 5:
            try:
                x_polish, val_polish = _high_precision_polish(best_x, max_time_budget=15.0)
                if val_polish < best_val:
                    best_val = val_polish
                    best_x = x_polish.copy()
            except Exception:
                pass
        
        # Final Safety Check
        if not np.isfinite(best_val) or best_val >= 1e10:
            best_x = seed
            best_val = _objective(seed)
        
        return np.array([float(best_x[0]), float(best_x[1]), float(best_x[2])])
        
    except Exception:
        seed = _seed()
        return np.array([float(seed[0]), float(seed[1]), float(seed[2])])
# EVOLVE-BLOCK-END
