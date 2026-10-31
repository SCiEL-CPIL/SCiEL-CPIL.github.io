"""Circle packing in perimeter-4 rectangle: Smooth-penalty SLSQP with soft-min boundaries."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # Safety margin before 150s kill

def _soft_min(vals, k=100.0):
    """
    Smooth approximation of the minimum of vals.
    vals: array of values
    k: sharpness parameter (higher is sharper)
    Returns: scalar soft-min
    """
    # log-sum-exp trick for stability: -1/k * log(sum(exp(-k * vals)))
    # To avoid overflow, subtract min first? 
    # Actually, for soft min, we want: min(x) ~ -1/k log(sum(exp(-k x)))
    # Let m = min(vals). Then exp(-k * (vals - m)) is safe.
    m = np.min(vals)
    # If k is large, exp(-k * (vals - m)) will be 1 for min, small for others.
    # Sum will be >= 1.
    # log(sum) is safe.
    # result = m - (1/k) * log(sum(exp(-k * (vals - m))))
    # This is stable.
    exp_terms = np.exp(-k * (vals - m))
    log_sum = np.log(np.sum(exp_terms))
    return m - log_sum / k

def _objective_smooth(x, W, H, mu=1e4, k=50.0):
    """
    Objective: minimize -sum(r) + mu * penalties.
    Uses soft-min for boundary constraints to ensure smooth gradients.
    x: (N, 3) array of [cx, cy, r]
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)

    # Clamp for stability
    cx = np.clip(cx, -0.5, W + 0.5)
    cy = np.clip(cy, -0.5, H + 0.5)
    r = np.clip(r, 1e-6, 1.0)

    val = -np.sum(r)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # 1. Boundary Constraints: r <= min(cx, cy, W-cx, H-cy)
    # We want to penalize r - soft_min(cx, cy, W-cx, H-cy) > 0
    # Let S_i = soft_min(cx_i, cy_i, W-cx_i, H-cy_i)
    # Penalty = max(r_i - S_i, 0)^2
    
    # Compute soft-min for all circles
    # dists: (N, 4)
    dists = np.column_stack([cx, cy, W - cx, H - cy])
    
    # Soft min calculation
    m = np.min(dists, axis=1)
    # exp(-k * (dists - m))
    # Shape (N, 4)
    exp_terms = np.exp(-k * (dists - m[:, np.newaxis]))
    log_sums = np.log(np.sum(exp_terms, axis=1))
    S = m - log_sums / k  # (N,)

    viol_b = np.maximum(r - S, 0.0)
    val += mu * np.sum(viol_b ** 2)
    
    # Gradient of boundary penalty
    # d/dx_i [ (r_i - S_i)^2 ] = 2 * (r_i - S_i) * (-dS_i/dx_i)
    # dS_i/dx_i = -1/k * sum_j [ -k * exp(-k(d_j - m)) / sum_exp ] * d(d_j - m)/dx_i
    # This is getting complex. Let's use the property that soft-min gradient is a weighted average of the gradients of the individual terms.
    # w_j = exp(-k(d_j - m)) / sum(exp(...))
    # dS/dx = - sum(w_j * d(d_j)/dx)
    # d(cx)/dx = 1, d(W-cx)/dx = -1, d(cy)/dx = 0, d(H-cy)/dx = 0
    # So dS/dcx = -(w_0 - w_2)
    # dS/dcy = -(w_1 - w_3)
    
    # Calculate weights (N, 4)
    weights = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)
    
    # dS/dcx = -(weights[:,0] - weights[:,2])
    # dS/dcy = -(weights[:,1] - weights[:,3])
    
    dS_dcx = -(weights[:, 0] - weights[:, 2])
    dS_dcy = -(weights[:, 1] - weights[:, 3])
    
    for i in range(n):
        if viol_b[i] > 0:
            dv = 2.0 * mu * viol_b[i]
            # grad_i = dv * (-dS_i/dx_i)
            grad[i, 0] += dv * (-dS_dcx[i])
            grad[i, 1] += dv * (-dS_dcy[i])
            grad[i, 2] += dv * 1.0

    # 2. Non-overlap Constraints
    # Penalty = max(r_i + r_j - dist, 0)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            req = r[i] + r[j]
            viol = req - d
            if viol <= 0:
                continue
            
            val += mu * viol ** 2
            
            if d < 1e-8:
                # Degenerate: push apart randomly or along axis
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * -1.0
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            
            dv = 2.0 * mu * viol
            # d viol / d cx_i = -dx/d
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    # Strict clamping for evaluation
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-9 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _grid_layout(kx, ky, W, H, shrink=0.99):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _force_relax(x0, W, H, steps=200, lr=0.01):
    """
    Force-directed relaxation to resolve overlaps and push circles to boundaries.
    This creates a valid starting point for SLSQP.
    """
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    n = len(cx)
    
    for step in range(steps):
        fcx = np.zeros(n)
        fcy = np.zeros(n)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.hypot(dx, dy)
                req = r[i] + r[j]
                if d < req and d > 1e-8:
                    f = (req - d) / d
                    fcx[i] += f * dx
                    fcy[i] += f * dy
                    fcx[j] -= f * dx
                    fcy[j] -= f * dy
                elif d <= 1e-8:
                    angle = random.uniform(0, 2 * np.pi)
                    fcx[i] += 0.1 * np.cos(angle)
                    fcy[i] += 0.1 * np.sin(angle)
                    fcx[j] -= 0.1 * np.cos(angle)
                    fcy[j] -= 0.1 * np.sin(angle)
        
        # Attraction to boundaries (grow radii)
        # We simply try to increase r slightly if it doesn't violate constraints
        # But in force relax, we just move centers to resolve overlap.
        # Then we grow r.
        
        cx += lr * fcx
        cy += lr * fcy
        
        # Keep centers in box
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        
        # Grow radii
        lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
        r = np.minimum(r * 1.001 + 0.0001, lo)
        r = np.maximum(r, 1e-6)
        
        lr *= 0.99
        
        # Check validity
        if _evaluate(np.column_stack([cx, cy, r]), W, H) is not None:
             # If valid, we can try to grow more aggressively?
             # No, let SLSQP do the fine tuning.
             pass
             
    return np.column_stack([cx, cy, r])

def _optimize_from(x0, W, H, max_iter=300, ftol=1e-10):
    """Run SLSQP with smooth objective."""
    best_x = x0.copy()
    best_sum = _evaluate(x0, W, H)
    if best_sum is None:
        best_sum = 0.0

    def fun(x):
        v, _ = _objective_smooth(x, W, H, mu=1e4, k=50.0)
        return v

    def grad(x):
        _, g = _objective_smooth(x, W, H, mu=1e4, k=50.0)
        return g

    try:
        res = minimize(fun, x0, jac=grad, method='SLSQP',
                       options={'maxiter': max_iter, 'ftol': ftol, 'disp': 0})
        if res.x is not None:
            s = _evaluate(res.x, W, H)
            if s is not None and s > best_sum:
                best_sum = s
                best_x = res.x
    except Exception:
        pass

    # Polish pass with tighter tolerance and higher mu/k for precision
    if time.time() < DEADLINE - 10:
        def fun_polish(x):
            v, _ = _objective_smooth(x, W, H, mu=1e5, k=100.0)
            return v

        def grad_polish(x):
            _, g = _objective_smooth(x, W, H, mu=1e5, k=100.0)
            return g

        try:
            res2 = minimize(fun_polish, best_x, jac=grad_polish, method='SLSQP',
                            options={'maxiter': 100, 'ftol': 1e-12, 'disp': 0})
            if res2.x is not None:
                s = _evaluate(res2.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_x = res2.x
        except Exception:
            pass

    return best_x, best_sum

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0

    def time_ok():
        return time.time() < DEADLINE

    random.seed(42)
    np.random.seed(42)

    candidates = []

    # Aspect ratios to try
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5), (5, 4), (4, 5), (6, 3), (3, 6)]

    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.95)
            # Relax to ensure validity and better start
            if time.time() > DEADLINE - 5:
                 return _grid_layout(5, 5, 1.0, 1.0, 0.9)
            g_relaxed = _force_relax(g, W, H, steps=50, lr=0.005)
            s = _evaluate(g_relaxed, W, H)
            if s is not None:
                candidates.append((s, W, H, g_relaxed))

    # Sort by initial score
    candidates.sort(key=lambda t: -t[0])

    # Optimize top candidates
    top_k = min(8, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        
        # Run optimization
        x_opt, s_opt = _optimize_from(g0, W, H, max_iter=150)
        
        if s_opt > best_sum:
            best_sum = s_opt
            best_arr = x_opt
            best_W = W
            best_H = H

    # If time remains, try random restarts from best
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(3, int(remaining // 20))
        for _ in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.01
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            
            # Quick relax to fix any overlaps introduced by noise
            x0_relaxed = _force_relax(x0, best_W, best_H, steps=20, lr=0.002)
            
            x_opt, s_opt = _optimize_from(x0_relaxed, best_W, best_H, max_iter=100)
            if s_opt > best_sum:
                best_sum = s_opt
                best_arr = x_opt

    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.9)
        best_arr = g
        best_W, best_H = W, H

    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])

    # Final validation
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Fallback to grid
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.9)
        return g

    return out

