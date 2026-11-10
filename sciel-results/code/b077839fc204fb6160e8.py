import numpy as np
from scipy import optimize
import time

_N = 8
_FRAC = 1.0 / _N
_IDX8 = np.arange(_N) / _N

# High resolution grid for c to accurately compute the integral
_C_GRID = np.linspace(0.0, 1.0, 20001)

def _union_len(C, base):
    """
    Compute the union length of intervals [base[i] + C*c_i, base[i] + C*c_i + 1/N]
    for each c in C.
    Returns an array of union lengths.
    Vectorized sweep-line: sort by left endpoint, then compute coverage via prefix max of right endpoints.
    """
    shifts = C[:, None] * _IDX8[None, :]
    l = base[None, :] + shifts
    r = l + _FRAC
    # Sort intervals by left endpoint
    order = np.argsort(l, axis=1)
    ls = np.take_along_axis(l, order, 1)
    rs = np.take_along_axis(r, order, 1)
    
    # Union length = sum over k of max(0, rs[k] - max(ls[k], max(rs[0..k-1])))
    # For k=0: contribution is rs[0] - ls[0]
    # For k>=1: contribution is max(0, rs[k] - max(ls[k], running_max_r))
    
    # Compute running max of right endpoints up to k-1
    # running_max_r[k] = max(rs[0], rs[1], ..., rs[k-1])
    # For k=0, there's no prior, so we handle separately.
    
    # Use cumulative max shifted by 1
    cummax_r = np.maximum.accumulate(rs, axis=1)
    # shifted: for position k, we want max of rs[0..k-1]
    # cummax_r[:, 0] = rs[:, 0]
    # We want for k>=1: cummax_r[:, k-1]
    # For k=0: use ls[:, 0] (so contribution = rs[0]-ls[0])
    
    # Build an array where at position k, we have the effective left bound
    # eff_left[k] = max(ls[k], cummax_r[k-1]) for k>=1, and ls[0] for k=0
    eff_left = np.empty_like(ls)
    eff_left[:, 0] = ls[:, 0]
    eff_left[:, 1:] = np.maximum(ls[:, 1:], cummax_r[:, :-1])
    
    # Contribution of each interval: max(0, rs - eff_left)
    contrib = np.maximum(0.0, rs - eff_left)
    union = contrib.sum(axis=1)
    return union

def _v_to_fg(v):
    """
    v has length 14: [f_1..f_7, g_1..g_7]
    f_0 = 0, g_0 = 0
    """
    f = np.zeros(_N)
    g = np.zeros(_N)
    f[1:] = v[:7]
    g[1:] = v[7:]
    return f, g

def _obj(v):
    """
    Exact 3-D union volume for the separable (f, g) ansatz.
    V = integral_0^1 Lx(c) * Ly(c) dc
    """
    f, g = _v_to_fg(v)
    uf = _union_len(_C_GRID, f)
    ug = _union_len(_C_GRID, g)
    
    # Use trapezoidal rule for integration
    val = float(np.trapz(uf * ug, _C_GRID))
    if not np.isfinite(val) or val <= 0:
        return 1e9
    return val

def _obj_grad(v):
    """
    Compute gradient of the volume objective using finite differences.
    Uses central differences with adaptive step size.
    """
    eps = 1e-6
    grad = np.zeros_like(v)
    for i in range(len(v)):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += eps
        v_minus[i] -= eps
        fp = _obj(v_plus)
        fm = _obj(v_minus)
        grad[i] = (fp - fm) / (2 * eps)
    return grad

def kakeya_needle_3d8() -> np.ndarray:
    N = _N
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0
    
    def build(f, g):
        i = np.arange(N * N) // N
        j = np.arange(N * N) % N
        return np.stack([f[i], g[j]], axis=1)

    # Initial guesses
    # 1. Zero (all at origin) - bad, volume 1
    # 2. Linear shear
    # 3. Random
    
    best_v = None
    best_val = 1e9
    
    # Helper to update best
    def consider(v_candidate):
        nonlocal best_v, best_val
        if v_candidate is None:
            return
        val = _obj(v_candidate)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_v = np.asarray(v_candidate, dtype=float).copy()
            # print(f"New best: {val}") # Disable for speed
    
    bounds = [(-3.0, 3.0)] * 14
    
    # 1. Try a few good initializations
    # Keich-like or structured
    # f(i) = -i/N is a counter-shear
    cs_f = -np.arange(1, N) / N
    cs_g = -np.arange(1, N) / N
    init_cs = np.concatenate([cs_f, cs_g])
    
    ps_f = np.arange(1, N) / N
    ps_g = np.arange(1, N) / N
    init_ps = np.concatenate([ps_f, ps_g])
    
    zero_init = np.zeros(14)
    
    for init in [init_cs, init_ps, zero_init]:
        if time.time() > start_time + time_limit - safety_margin:
            break
        try:
            # Use gradient for faster convergence with smooth objective
            res = optimize.minimize(_obj, init, jac=_obj_grad, method='L-BFGS-B', 
                                    bounds=bounds, 
                                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.success:
                consider(res.x)
        except Exception:
            pass

    # 2. Global Search using Differential Evolution
    # DE is robust for global optimization
    if time.time() < start_time + time_limit - 60.0:
        try:
            de_res = optimize.differential_evolution(
                _obj, 
                bounds=bounds, 
                seed=42, 
                maxiter=500, 
                tol=1e-10,
                popsize=15,
                polish=True # Polishes the best result
            )
            if de_res.success:
                consider(de_res.x)
            else:
                consider(de_res.x) # Still take it
        except Exception:
            pass

    # 3. Further local refinement and multi-start
    if best_v is not None:
        if time.time() < start_time + time_limit - 30.0:
            try:
                res = optimize.minimize(_obj, best_v, jac=_obj_grad, method='L-BFGS-B', 
                                        bounds=bounds,
                                        options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-14})
                consider(res.x)
            except Exception:
                pass

    # 4. Random restarts if time remains
    rng = np.random.default_rng(12345)
    while time.time() < start_time + time_limit - 10.0:
        v0 = rng.uniform(-1.5, 1.5, 14)
        try:
            # Quick local search
            res = optimize.minimize(_obj, v0, method='Nelder-Mead', 
                                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success:
                consider(res.x)
        except Exception:
            pass

    # Construct final result
    if best_v is None:
        # Fallback
        f = np.zeros(N)
        g = np.zeros(N)
    else:
        f, g = _v_to_fg(best_v)

    pos = build(f, g)
    
    # Sanity check
    if pos.shape != (N * N, 2) or not np.all(np.isfinite(pos)):
        f = np.zeros(N)
        g = np.zeros(N)
        pos = build(f, g)
        
    return pos
