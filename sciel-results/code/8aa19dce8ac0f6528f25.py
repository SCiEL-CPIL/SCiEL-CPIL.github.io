"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_R(h):
    n = len(h)
    R = np.empty(n-1)
    for k in range(1, n):
        R[k-1] = np.dot(h[:n-k], h[k:]) * (2.0/n)
    return R

def _min_R(h):
    R = _compute_R(h)
    return np.min(R), int(np.argmin(R))

def _grad_R_k(h, k):
    n = len(h)
    g = np.zeros(n)
    c = 2.0/n
    g[:n-k] += c * h[k:]
    g[k:] += c * h[:n-k]
    return g

def _project(v, target_sum):
    n = len(v)
    x = np.clip(v, 0.0, 1.0)
    for _ in range(100):
        s = np.sum(x)
        diff = target_sum - s
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = x < 1.0 - 1e-12
            cnt = np.sum(mask)
            if cnt == 0:
                break
            x[mask] += diff/cnt
        else:
            mask = x > 1e-12
            cnt = np.sum(mask)
            if cnt == 0:
                break
            x[mask] += diff/cnt
        x = np.clip(x, 0.0, 1.0)
    return x

def _soft_min_obj_grad(x, beta):
    n = len(x)
    R = _compute_R(x)
    min_val = np.min(R)
    shifted = R - min_val
    exp_neg = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_neg)
    soft_min = min_val - np.log(sum_exp)/beta
    obj = 1.0 - soft_min
    weights = exp_neg / sum_exp
    g = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0/n)
        if w > 1e-15:
            g[:n-k] += w * x[k:]
            g[k:] += w * x[:n-k]
    return obj, -g

def _penalty_obj_grad(x, t, mu):
    n = len(x)
    R = _compute_R(x)
    deficit = np.maximum(0.0, t - R)
    obj = mu * np.sum(deficit**2)
    g = np.zeros(n)
    for k in range(1, n):
        d = t - R[k-1]
        if d > 0:
            gk = _grad_R_k(x, k)
            g -= 2.0 * mu * d * gk
    return obj, g

def _boost_min(h, target_sum, n_steps=2000, step_init=0.05, rng_seed=None):
    n = len(h)
    h = _project(h.copy(), target_sum)
    best_min = _min_R(h)[0]
    best_h = h.copy()
    step = step_init
    rng = np.random.RandomState(rng_seed) if rng_seed is not None else None
    for it in range(n_steps):
        if _time_remaining() < 5.0:
            break
        R = _compute_R(h)
        min_val = np.min(R)
        k_star = int(np.argmin(R)) + 1
        if min_val > best_min + 1e-12:
            best_min = min_val
            best_h = h.copy()
        g = _grad_R_k(h, k_star)
        # Add small random perturbation to escape local minima
        if rng is not None and it % 100 == 50:
            g += 0.01 * rng.normal(0, 1, n)
        h_new = h + step * g
        h_new = _project(h_new, target_sum)
        new_min = _min_R(h_new)[0]
        if new_min >= min_val - 1e-12:
            h = h_new
        else:
            step *= 0.85
            if step < 1e-8:
                break
    return best_h, best_min

def _generate_initials(n, target_sum):
    inits = []
    inits.append(np.full(n, 0.5))
    inits.append(np.array([0.0 if i%2==0 else 1.0 for i in range(n)]))
    inits.append(np.array([1.0 if i%2==0 else 0.0 for i in range(n)]))
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n*frac)
        if 0 < k < n:
            a = target_sum/k
            b = (target_sum - k*a)/(n-k)
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
            inits.append(np.concatenate([np.full(k, a), np.full(n-k, b)]))
            inits.append(np.concatenate([np.full(n-k, b), np.full(k, a)]))
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for amp in [0.15, 0.25, 0.35]:
            inits.append(0.5 + amp*np.sin(freq*x))
            inits.append(0.5 + amp*np.cos(freq*x))
    tri = np.array([abs(2*(i/n - 0.5)) for i in range(n)])
    tri = (tri - tri.mean())/(tri.std()+1e-10)
    inits.append(0.5 + 0.3*tri)
    rng = np.random.RandomState(42)
    for _ in range(25):
        inits.append(rng.uniform(0, 1, n))
    inits = [_project(x, target_sum) for x in inits]
    return inits

def construct():
    n = N
    target_sum = n/2.0
    best_h = np.full(n, 0.5)
    best_min_R = _min_R(best_h)[0]
    inits = _generate_initials(n, target_sum)
    
    # Phase 1: Soft-min L-BFGS on diverse initials
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    for x0 in inits:
        if _time_remaining() < 10.0:
            break
        x = x0.copy()
        for beta in beta_schedule:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(_soft_min_obj_grad, x, args=(beta,),
                    method='L-BFGS-B', jac=True, bounds=[(0,1)]*n,
                    options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-13})
                res_x = _project(res.x, target_sum)
                min_R_val, _ = _min_R(res_x)
                if min_R_val > best_min_R:
                    best_min_R = min_R_val
                    best_h = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: Penalty-based threshold raising
    if _time_remaining() > 35.0:
        h = best_h.copy()
        t = best_min_R
        for _ in range(40):
            if _time_remaining() < 8.0:
                break
            t += 0.004
            try:
                res = minimize(_penalty_obj_grad, h, args=(t, 1.0),
                    method='L-BFGS-B', jac=True, bounds=[(0,1)]*n,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
                res_x = _project(res.x, target_sum)
                min_R_val, _ = _min_R(res_x)
                if min_R_val > best_min_R:
                    best_min_R = min_R_val
                    best_h = res_x
                h = res_x
            except Exception:
                pass
    
    # Phase 3: Boost-min with random restarts
    if _time_remaining() > 25.0:
        for seed in [None, 1, 2, 3]:
            if _time_remaining() < 8.0:
                break
            h_boost, min_boost = _boost_min(best_h, target_sum, n_steps=2000, step_init=0.05, rng_seed=seed)
            if min_boost > best_min_R:
                best_min_R = min_boost
                best_h = h_boost
    
    # Phase 4: High-beta polish
    if _time_remaining() > 15.0:
        for beta in [300.0, 500.0, 1000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(_soft_min_obj_grad, best_h, args=(beta,),
                    method='L-BFGS-B', jac=True, bounds=[(0,1)]*n,
                    options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14})
                res_x = _project(res.x, target_sum)
                min_R_val, _ = _min_R(res_x)
                if min_R_val > best_min_R:
                    best_min_R = min_R_val
                    best_h = res_x
            except Exception:
                pass
    
    # Phase 5: Final fine boost
    if _time_remaining() > 8.0:
        h_boost, min_boost = _boost_min(best_h, target_sum, n_steps=1500, step_init=0.02)
        if min_boost > best_min_R:
            best_min_R = min_boost
            best_h = h_boost
    
    best_h = np.clip(best_h, 0, 1)
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-10) & (best_h < 1-1e-10)
        if np.sum(mask) > 0:
            best_h[mask] += diff/np.sum(mask)
            best_h = np.clip(best_h, 0, 1)
    return best_h
# EVOLVE-BLOCK-END
