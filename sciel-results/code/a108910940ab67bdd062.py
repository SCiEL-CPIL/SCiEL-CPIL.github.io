import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 145.0  # Safety margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _objective(params, start, beta=50.0, lambda_eq=0.0):
    """
    Objective: maximize sum of radii with smooth penalty for violations.
    Uses log-sum-exp for smooth max.
    """
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = np.maximum(params[2*N:3*N], 0.0)
    
    obj = -np.sum(r)
    
    # Pairwise constraints
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    violations = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    v_pairs = violations[mask]
    
    v_max = np.max(v_pairs)
    if v_max > 0:
        lse = v_max + np.log(np.sum(np.exp(beta * (v_pairs - v_max)))) / beta
    else:
        lse = 0.0
    
    # Boundary constraints
    bviol = np.maximum(0, r - x) + np.maximum(0, x - (1.0 - r)) + \
            np.maximum(0, r - y) + np.maximum(0, y - (1.0 - r))
    b_max = np.max(bviol)
    if b_max > 0:
        lse_b = b_max + np.log(np.sum(np.exp(beta * (bviol - b_max)))) / beta
    else:
        lse_b = 0.0
    
    mu = 500.0
    result = obj + mu * (lse + lse_b)
    
    if lambda_eq > 0:
        r_mean = np.mean(r)
        r_var = np.sum((r - r_mean)**2)
        result += lambda_eq * r_var
        
    return result

def _objective_grad(params, start, beta=50.0, lambda_eq=0.0):
    """Gradient of the objective function."""
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = np.maximum(params[2*N:3*N].copy(), 0.0)
    
    grad = np.zeros(3*N)
    grad[2*N:3*N] = -1.0
    
    # Pairwise constraints gradient
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    violations = min_dist - dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    v_pairs = violations[mask]
    
    if np.max(v_pairs) > 0:
        v_max = np.max(v_pairs)
        exps = np.exp(beta * (v_pairs - v_max))
        weights = exps / np.sum(exps)
        idx_i, idx_j = np.where(mask)
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            w = weights[k]
            if w < 1e-12:
                continue
            d = dist[i, j]
            if d < 1e-12:
                continue
            gx = -(dx[i, j] / d)
            gy = -(dy[i, j] / d)
            grad[i] += w * gx
            grad[N + i] += w * gy
            grad[2*N + i] += w
            grad[j] += w * (-gx)
            grad[N + j] += w * (-gy)
            grad[2*N + j] += w
            
    # Boundary constraints gradient
    bviol = np.maximum(0, r - x) + np.maximum(0, x - (1.0 - r)) + \
            np.maximum(0, r - y) + np.maximum(0, y - (1.0 - r))
    b_max = np.max(bviol)
    if b_max > 0:
        exps = np.exp(beta * (bviol - b_max))
        weights = exps / np.sum(exps)
        
        for i in range(N):
            w = weights[i]
            if w < 1e-12:
                continue
            if r[i] > x[i]:
                grad[i] += -w
            if x[i] > 1.0 - r[i]:
                grad[i] += w
            if r[i] > y[i]:
                grad[N+i] += -w
            if y[i] > 1.0 - r[i]:
                grad[N+i] += w
            if r[i] > x[i]:
                grad[2*N+i] += w
            if x[i] > 1.0 - r[i]:
                grad[2*N+i] += w
            if r[i] > y[i]:
                grad[2*N+i] += w
            if y[i] > 1.0 - r[i]:
                grad[2*N+i] += w
                
    if lambda_eq > 0:
        r_mean = np.mean(r)
        grad[2*N:3*N] += lambda_eq * 2 * (r - r_mean)
        
    mu = 500.0
    grad = grad * mu
    return grad

def _make_valid(x, y, r):
    """Ensure the configuration is valid by shrinking radii and pushing apart."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(50):
        # Push apart
        for _iter in range(100):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.hypot(dx, dy)
                    need = r[i] + r[j]
                    if d < need:
                        if d < 1e-9:
                            ang = np.random.uniform(0, 2*np.pi)
                            ux, uy = np.cos(ang), np.sin(ang)
                        else:
                            ux, uy = dx/d, dy/d
                        gap = need - d
                        step = 0.5 * gap * 0.5
                        x[i] += step * ux
                        y[i] += step * uy
                        x[j] -= step * ux
                        y[j] -= step * uy
                        moved = True
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            if not moved:
                break
        
        if _is_valid(x, y, r):
            break
            
        # Shrink radii
        for i in range(N):
            max_r = min(x[i], 1-x[i], y[i], 1-y[i])
            for j in range(N):
                if i != j:
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    max_r = min(max_r, (d - r[j]) * 0.99)
            if r[i] > max_r:
                r[i] = max(0.0, max_r)
                
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.05)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _grid_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.9)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _phyllotaxis_init():
    phi = np.pi * (3.0 - np.sqrt(5.0))
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        r = np.sqrt((i + 0.5) / N) * 0.45
        theta = i * phi
        x[i] = 0.5 + r * np.cos(theta)
        y[i] = 0.5 + r * np.sin(theta)
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.4 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _optimize_with_growth(x, y, r, start, max_growth_steps=5):
    """
    Optimize positions/radii, then grow radii, then optimize again.
    """
    best = (x, y, r)
    best_score = _sum_r(r)
    
    for step in range(max_growth_steps):
        if _elapsed(start) > TIME_LIMIT - 10:
            break
            
        # 1. Optimize current configuration
        p0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.001, 0.45))
            
        try:
            res = minimize(
                _objective, p0, args=(start, 50.0, 0.0),
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-6}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
                x, y, r = vx, vy, vr
        except Exception:
            pass
            
        # 2. Grow radii
        # Calculate max possible growth for each circle
        growths = []
        for i in range(N):
            slack_b = min(x[i]-r[i], 1-x[i]-r[i], y[i]-r[i], 1-y[i]-r[i])
            slack_p = 1e9
            for j in range(N):
                if i != j:
                    d = np.hypot(x[i]-x[j], y[i]-y[j])
                    slack_p = min(slack_p, d - r[i] - r[j])
            max_g = min(slack_b, slack_p)
            growths.append(max(0.0, max_g))
            
        min_g = min(growths)
        if min_g < 1e-6:
            break
            
        # Grow by 50% of the minimum slack to maintain balance
        grow_amt = 0.5 * min_g
        r = r + grow_amt
        
        # Ensure validity after growth
        x, y, r = _make_valid(x, y, r)
        
    return best

def construct():
    start = time.time()
    
    inits = [
        _phyllotaxis_init(),
        _grid_init(6, 6),
        _grid_init(5, 7),
        _random_init(1),
        _random_init(2),
        _random_init(3),
        _random_init(42),
        _random_init(100)
    ]
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])
        
    for k, init in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 20:
            break
        x, y, r = init
        if not _is_valid(x, y, r):
            x, y, r = _make_valid(x, y, r)
            
        res = _optimize_with_growth(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res
                best_score = s
                
    # Final polish on the best
    if _elapsed(start) < TIME_LIMIT - 5:
        x, y, r = best
        p0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.001, 0.45))
            
        try:
            res = minimize(
                _objective, p0, args=(start, 100.0, 10.0),
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best = (vx, vy, vr)
                    best_score = s
        except Exception:
            pass
            
    x, y, r = best
    result = np.column_stack((x, y, r))
    return result
