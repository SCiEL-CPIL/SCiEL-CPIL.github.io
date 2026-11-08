"""Circle packing: 21 circles in perimeter-4 rectangle, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before 150s kill
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _validate(x, L, W):
    """Return sum of radii if valid, else -inf. x: (n,3) centers+radius."""
    cx = x[:, 0]; cy = x[:, 1]; r = x[:, 2]
    if np.any(r <= 0):
        return -np.inf
    if np.any(cx < r) or np.any(cy < r):
        return -np.inf
    if np.any(cx > L - r) or np.any(cy > W - r):
        return -np.inf
    # pairwise
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            if d < r[i]+r[j] - 1e-9:
                return -np.inf
    return float(np.sum(r))

def _objective(params, L, W):
    """Objective for optimizer: maximize sum of radii (return negative)."""
    x = params.reshape(N, 3)
    cx = x[:, 0]; cy = x[:, 1]; r = x[:, 2]
    # penalty for boundary
    pen = 0.0
    for i in range(N):
        pen += max(0.0, r[i] - cx[i])**2
        pen += max(0.0, r[i] - cy[i])**2
        pen += max(0.0, r[i] - (L - cx[i]))**2
        pen += max(0.0, r[i] - (W - cy[i]))**2
    # pairwise
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(cx[i]-cx[j], cy[i]-cy[j])
            pen += max(0.0, r[i]+r[j]-d)**2
    return -float(np.sum(r)) + 1e4 * pen

def _gradient(params, L, W):
    """Subgradient of the penalized objective."""
    x = params.reshape(N, 3)
    cx = x[:, 0]; cy = x[:, 1]; r = x[:, 2]
    grad = np.zeros_like(params)
    # sum of radii: grad w.r.t. r is -1
    grad[2::3] = -1.0
    # boundary penalties
    for i in range(N):
        # left: r - cx
        v = r[i] - cx[i]
        if v > 0:
            grad[3*i]   += 2e4 * (-v)   # d/d cx
            grad[3*i+2] += 2e4 * ( v)   # d/d r
        # bottom: r - cy
        v = r[i] - cy[i]
        if v > 0:
            grad[3*i+1] += 2e4 * (-v)
            grad[3*i+2] += 2e4 * ( v)
        # right: r - (L - cx) = r + cx - L
        v = r[i] + cx[i] - L
        if v > 0:
            grad[3*i]   += 2e4 * ( v)
            grad[3*i+2] += 2e4 * ( v)
        # top: r + cy - W
        v = r[i] + cy[i] - W
        if v > 0:
            grad[3*i+1] += 2e4 * ( v)
            grad[3*i+2] += 2e4 * ( v)
    # pairwise
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i]-cx[j]; dy = cy[i]-cy[j]
            d = np.hypot(dx, dy)
            v = r[i]+r[j]-d
            if v > 0 and d > 1e-12:
                gx = 2e4 * v * (dx/d)
                gy = 2e4 * v * (dy/d)
                grad[3*i]   += gx
                grad[3*i+1] += gy
                grad[3*i+2] += 2e4 * v
                grad[3*j+2] += 2e4 * v
                grad[3*j]   -= gx
                grad[3*j+1] -= gy
    return grad

def _polish(x, L, W, max_iter=200, tol=1e-9):
    """Local polish with L-BFGS-B."""
    params = x.flatten()
    try:
        res = minimize(_objective, params, args=(L, W), jac=_gradient,
                       method='L-BFGS-B', options={'maxiter': max_iter,
                       'ftol': tol, 'gtol': tol, 'maxls': 50})
        new_x = res.x.reshape(N, 3)
        # clip tiny negatives
        new_x[:, 2] = np.maximum(new_x[:, 2], 1e-6)
        # project to feasible region
        new_x[:, 0] = np.clip(new_x[:, 0], new_x[:, 2], L - new_x[:, 2])
        new_x[:, 1] = np.clip(new_x[:, 1], new_x[:, 2], W - new_x[:, 2])
        val = _validate(new_x, L, W)
        if val > _validate(x, L, W):
            return new_x, val
    except Exception:
        pass
    return x, _validate(x, L, W)

def _sa(x, L, W, iters=3000, T0=0.05, T1=1e-4, seed=0):
    """Simulated annealing on centers+radius."""
    rng = np.random.default_rng(seed)
    cur = x.copy()
    cur_val = _validate(cur, L, W)
    best = cur.copy()
    best_val = cur_val
    for it in range(iters):
        if _time_left() < 5:
            break
        # propose move
        idx = rng.integers(N)
        move = rng.normal(0, T0, size=2)
        dr = rng.normal(0, T0 * 0.5)
        new = cur.copy()
        new[idx, 0] = np.clip(cur[idx, 0] + move[0], 0.01, L - 0.01)
        new[idx, 1] = np.clip(cur[idx, 1] + move[1], 0.01, W - 0.01)
        new[idx, 2] = max(0.01, cur[idx, 2] + dr)
        # clip to boundary
        new[idx, 2] = min(new[idx, 2], new[idx, 0], new[idx, 1], L - new[idx, 0], W - new[idx, 1])
        new_val = _validate(new, L, W)
        T = T0 * (T1/T0) ** (it / max(1, iters))
        if new_val > cur_val or (new_val > -np.inf and rng.random() < np.exp((new_val - cur_val) / max(T, 1e-9))):
            cur = new
            cur_val = new_val
            if cur_val > best_val:
                best = cur.copy()
                best_val = cur_val
    return best, best_val

def _grid_init(L, W):
    """Initial grid placement."""
    cols = int(np.ceil(np.sqrt(N * L / W))) if L >= W else int(np.ceil(np.sqrt(N * W / L)))
    rows = int(np.ceil(N / cols))
    x = np.zeros((N, 3))
    for i in range(N):
        ci = i % cols
        ri = i // cols
        x[i, 0] = (ci + 0.5) * L / cols
        x[i, 1] = (ri + 0.5) * W / rows
        x[i, 2] = min(L/cols, W/rows) * 0.45
    return x

def construct():
    global _START
    _START = time.time()
    best_x = None
    best_val = -np.inf
    best_LW = (1.0, 1.0)
    
    # Try a range of aspect ratios
    # Perimeter = 2(L+W) = 4 => L+W = 2
    # L in (0, 2), W = 2 - L
    aspect_ratios = []
    # Square and near-square
    aspect_ratios.append((1.0, 1.0))
    # Elongated
    for L in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
        W = 2.0 - L
        if W > 0.1:
            aspect_ratios.append((L, W))
    
    for L, W in aspect_ratios:
        if _time_left() < 10:
            break
        # Multiple seeds
        for seed in [0, 1, 2]:
            if _time_left() < 5:
                break
            x = _grid_init(L, W)
            # SA
            x, val = _sa(x, L, W, iters=2000, seed=seed)
            # Polish
            x, val = _polish(x, L, W, max_iter=100)
            if val > best_val:
                best_val = val
                best_x = x.copy()
                best_LW = (L, W)
    
    # Final polish with more iterations
    if best_x is not None and _time_left() > 10:
        L, W = best_LW
        best_x, best_val = _polish(best_x, L, W, max_iter=500, tol=1e-10)
    
    if best_x is None:
        # Fallback: simple grid in square
        x = _grid_init(1.0, 1.0)
        best_x = x
        best_val = _validate(x, 1.0, 1.0)
    
    return best_x

# EVOLVE-BLOCK-END
