import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # Safety margin under 150s
X_GRID = np.linspace(-0.5, 0.5, N)  # step centers on [-1/4, 1/4] mapped to [-0.5, 0.5] for convenience

def _value_from_h(h):
    """
    Computes the value as per the contract.
    h: array of 50 non-negative heights.
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)) or np.any(h < 0):
        return 0.0
    
    mx = np.max(h)
    if mx < 1e-12:
        return 0.0
    h = h / mx
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    a = y[:-1]
    b = y[1:]
    l2 = (w / 3.0) * np.sum(a * a + a * b + b * b)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    try:
        h = np.clip(h, 0.0, 1.0)
        return -_value_from_h(h)
    except Exception:
        return 1e6

def _params_to_h(params):
    """
    Map parametric representation to 50-dim height vector.
    params: array of shape (3*K,) where K = number of Gaussians
    Each triple: (center, width, amplitude)
    Centers in [-0.4, 0.4], widths in [0.05, 0.3], amplitudes in [0.1, 1.0]
    """
    K = len(params) // 3
    h = np.zeros(N)
    for k in range(K):
        c = params[3*k]
        w = params[3*k+1]
        a = params[3*k+2]
        h += a * np.exp(-((X_GRID - c) ** 2) / (2.0 * w ** 2))
    mx = np.max(h)
    if mx > 1e-12:
        h = h / mx
    else:
        h = np.ones(N) / N
    return np.clip(h, 0.0, 1.0)

def _neg_value_params(params):
    try:
        h = _params_to_h(params)
        return -_value_from_h(h)
    except Exception:
        return 1e6

def _coordinate_descent(h, max_rounds=50, tol=1e-10, deadline=None):
    """
    Simple coordinate descent for polishing.
    """
    h = np.array(h, dtype=float)
    best_val = _value_from_h(h)
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old = h[i]
            # Try a few step sizes
            for step in [0.01, 0.05, 0.1, 0.2, -0.01, -0.05, -0.1, -0.2]:
                h_new = h.copy()
                h_new[i] = max(0.0, old + step)
                v = _value_from_h(h_new)
                if v > best_val:
                    h[i] = h_new[i]
                    best_val = v
                    improved = True
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N) / N
    best_val = _value_from_h(best_h)
    
    # Phase 1: Parametric global search using Differential Evolution
    # Use K=4 Gaussians -> 12 parameters
    K = 4
    n_params = 3 * K
    param_bounds = []
    for _ in range(K):
        param_bounds.append((-0.4, 0.4))   # center
        param_bounds.append((0.05, 0.3))   # width
        param_bounds.append((0.1, 1.0))    # amplitude
    
    # Run DE in parametric space
    if time.time() - start_time < DEADLINE - 60:
        try:
            res_de = differential_evolution(
                _neg_value_params,
                param_bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol'
            )
            if res_de.fun < 1e5:  # valid result
                h_de = _params_to_h(res_de.x)
                v_de = _value_from_h(h_de)
                if v_de > best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Coordinate descent polish on best_h
    if time.time() - start_time < DEADLINE - 20:
        h_cd, v_cd = _coordinate_descent(best_h, max_rounds=30, deadline=start_time + DEADLINE - 15)
        if v_cd > best_val:
            best_val = v_cd
            best_h = h_cd.copy()
    
    # Phase 3: L-BFGS-B full-dim polish
    bounds = [(0.0, 1.0) for _ in range(N)]
    if time.time() - start_time < DEADLINE - 10:
        try:
            res = minimize(_neg_value, best_h, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            h_res = np.clip(res.x, 0.0, 1.0)
            v_res = -res.fun
            if v_res > best_val:
                best_val = v_res
                best_h = h_res.copy()
        except Exception:
            pass
    
    # Phase 4: Try a few more parametric configurations with different K
    if time.time() - start_time < DEADLINE - 10:
        for K2 in [3, 5]:
            if time.time() - start_time > DEADLINE - 5:
                break
            n_p2 = 3 * K2
            pb2 = []
            for _ in range(K2):
                pb2.append((-0.4, 0.4))
                pb2.append((0.05, 0.3))
                pb2.append((0.1, 1.0))
            try:
                res2 = differential_evolution(
                    _neg_value_params,
                    pb2,
                    seed=42 + K2,
                    maxiter=40,
                    popsize=12,
                    tol=1e-6,
                    polish=False
                )
                if res2.fun < 1e5:
                    h2 = _params_to_h(res2.x)
                    # Polish this
                    h2, v2 = _coordinate_descent(h2, max_rounds=15, deadline=start_time + DEADLINE - 5)
                    if v2 > best_val:
                        best_val = v2
                        best_h = h2.copy()
                        # L-BFGS polish
                        if time.time() - start_time < DEADLINE - 3:
                            try:
                                res3 = minimize(_neg_value, best_h, method='L-BFGS-B', bounds=bounds,
                                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
                                h3 = np.clip(res3.x, 0.0, 1.0)
                                v3 = -res3.fun
                                if v3 > best_val:
                                    best_val = v3
                                    best_h = h3.copy()
                            except Exception:
                                pass
            except Exception:
                pass
    
    # Phase 5: Final random restarts with L-BFGS
    if time.time() - start_time < DEADLINE - 5:
        for i in range(5):
            if time.time() - start_time > DEADLINE - 3:
                break
            h_rand = rng.uniform(0, 1, N)
            try:
                res = minimize(_neg_value, h_rand, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                h_res = np.clip(res.x, 0.0, 1.0)
                v_res = -res.fun
                if v_res > best_val:
                    best_val = v_res
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Final check
    best_h = np.clip(best_h, 0.0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)

if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    t1 = time.time()
    v = _value_from_h(h)
    print(f"Time: {t1-t0:.2f}s, Value: {v:.6f}")
