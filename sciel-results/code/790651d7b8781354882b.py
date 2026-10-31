"""Circle packing in a perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin below 120s limit

def _is_valid(circles, w, h, tol=1e-5):
    n = len(circles)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]

    if np.any(rs < -tol):
        return False
    if np.any(xs < -rs - tol) or np.any(xs > w + rs + tol):
        return False
    if np.any(ys < -rs - tol) or np.any(ys > h + rs + tol):
        return False

    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx*dx + dy*dy < (rs[i] + rs[j])**2 - tol:
                return False
    return True

def _pebble_grow(x, y, W, H, r0=0.02, max_iter=2000, dt=0.001):
    """
    Pebble growth algorithm: start with small circles, grow them by
    the minimum distance to nearest neighbor or boundary.
    """
    xs = x.copy()
    ys = y.copy()
    rs = np.full(N, r0)
    
    for it in range(max_iter):
        # Compute minimum distance to boundary and neighbors for each circle
        min_dists = np.full(N, np.inf)
        
        for i in range(N):
            # Distance to boundaries
            d_left = xs[i] - rs[i]
            d_right = W - xs[i] - rs[i]
            d_bottom = ys[i] - rs[i]
            d_top = H - ys[i] - rs[i]
            min_dists[i] = min(d_left, d_right, d_bottom, d_top)
            
            # Distance to other circles
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_dists[i] = min(min_dists[i], d - rs[i] - rs[j])
        
        # Grow all circles by the minimum of their min_dists
        min_min = np.min(min_dists)
        if min_min <= 0:
            break
        
        growth = min(min_min, dt)
        rs += growth
        
        # If growth is too small, we've converged
        if growth < 1e-6:
            break
    
    return xs, ys, rs

def _force_relax(x, y, rs, W, H, max_iter=500, dt=0.002):
    """
    Force-directed relaxation to resolve overlaps and push circles
    away from boundaries.
    """
    xs = x.copy()
    ys = y.copy()
    r = rs.copy()
    
    for _ in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Wall repulsion
        for i in range(N):
            if xs[i] - r[i] < 0:
                fx[i] += (xs[i] - r[i]) * 50
            if xs[i] + r[i] > W:
                fx[i] -= (xs[i] + r[i] - W) * 50
            if ys[i] - r[i] < 0:
                fy[i] += (ys[i] - r[i]) * 50
            if ys[i] + r[i] > H:
                fy[i] -= (ys[i] + r[i] - H) * 50
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 20
                    fy[i] += (dy / d) * overlap * 20
                    fx[j] -= (dx / d) * overlap * 20
                    fy[j] -= (dy / d) * overlap * 20
        
        # Update positions
        xs += dt * fx
        ys += dt * fy
        
        # Clamp positions
        for i in range(N):
            xs[i] = np.clip(xs[i], r[i], W - r[i])
            ys[i] = np.clip(ys[i], r[i], H - r[i])
    
    return xs, ys

def _objective(z, mu):
    """
    z layout: [W, x1..xn, y1..yn, r1..rn]
    Returns -sum(r_i) + mu * penalty terms.
    """
    W = z[0]
    H = 2.0 - W
    xs = z[1:1+N]
    ys = z[1+N:1+2*N]
    rs = z[1+2*N:1+3*N]
    
    obj = -np.sum(rs)
    pen = 0.0
    
    # Boundary penalties
    v1 = np.maximum(rs - xs, 0.0)
    pen += np.sum(v1**2)
    v2 = np.maximum(xs + rs - W, 0.0)
    pen += np.sum(v2**2)
    v3 = np.maximum(rs - ys, 0.0)
    pen += np.sum(v3**2)
    v4 = np.maximum(ys + rs - H, 0.0)
    pen += np.sum(v4**2)
    
    # Overlap penalties
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx*dx + dy*dy
            rsum = rs[i] + rs[j]
            if d2 < rsum*rsum:
                pen += (rsum*rsum - d2)**2
    
    return obj + mu * pen

def _objective_grad(z, mu):
    """Gradient of the penalty objective."""
    W = z[0]
    H = 2.0 - W
    xs = z[1:1+N]
    ys = z[1+N:1+2*N]
    rs = z[1+2*N:1+3*N]
    
    grad = np.zeros(1 + 3*N)
    
    # Boundary gradients
    v1 = np.maximum(rs - xs, 0.0)
    grad[1:1+N] -= 2.0 * v1
    grad[1+2*N:1+3*N] += 2.0 * v1
    
    v2 = np.maximum(xs + rs - W, 0.0)
    grad[0] -= 2.0 * np.sum(v2)
    grad[1:1+N] += 2.0 * v2
    grad[1+2*N:1+3*N] += 2.0 * v2
    
    v3 = np.maximum(rs - ys, 0.0)
    grad[1+N:1+2*N] -= 2.0 * v3
    grad[1+2*N:1+3*N] += 2.0 * v3
    
    v4 = np.maximum(ys + rs - H, 0.0)
    grad[0] += 2.0 * np.sum(v4)
    grad[1+N:1+2*N] += 2.0 * v4
    grad[1+2*N:1+3*N] += 2.0 * v4
    
    # Overlap gradients
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx*dx + dy*dy
            rsum = rs[i] + rs[j]
            if d2 < rsum*rsum:
                diff = rsum*rsum - d2
                grad[1+i] += -4.0 * diff * dx
                grad[1+j] += 4.0 * diff * dx
                grad[1+N+i] += -4.0 * diff * dy
                grad[1+N+j] += 4.0 * diff * dy
                grad[1+2*N+i] += 4.0 * diff * rsum
                grad[1+2*N+j] += 4.0 * diff * rsum
    
    return mu * grad

def _extract_and_check(z):
    """Extract circles from optimization vector and check validity."""
    W = z[0]
    H = 2.0 - W
    xs = z[1:1+N]
    ys = z[1+N:1+2*N]
    rs = z[1+2*N:1+3*N]
    
    circles = np.column_stack([xs, ys, rs])
    
    # Clip to valid bounds
    circles[:, 0] = np.clip(circles[:, 0], 0, W)
    circles[:, 1] = np.clip(circles[:, 1], 0, H)
    circles[:, 2] = np.maximum(0.0, circles[:, 2])
    
    # Ensure boundary feasibility
    for i in range(N):
        max_r = min(circles[i, 0], circles[i, 1], W - circles[i, 0], H - circles[i, 1])
        circles[i, 2] = min(circles[i, 2], max_r)
    
    if _is_valid(circles, W, H):
        return circles, W, H, np.sum(circles[:, 2])
    return None, W, H, 0.0

def _grid_init(W, H, n_cols, n_rows):
    xs = (np.arange(n_cols) + 0.5) * (W / n_cols)
    ys = (np.arange(n_rows) + 0.5) * (H / n_rows)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X = X.flatten()
    Y = Y.flatten()
    X = X[:N]
    Y = Y[:N]
    r0 = min(W / n_cols, H / n_rows) / 2.0 * 0.9
    R = np.full(N, r0)
    return np.concatenate([[W], X, Y, R])

def _random_init(W, H, seed=None):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.05*W, 0.95*W, N)
    ys = rng.uniform(0.05*H, 0.95*H, N)
    rs = np.full(N, 0.08)
    return np.concatenate([[W], xs, ys, rs])

def _pebble_init(W, H, seed=None):
    """Initialize using pebble growth from random starting positions."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1*W, 0.9*W, N)
    ys = rng.uniform(0.1*H, 0.9*H, N)
    
    # Grow circles using pebble algorithm
    xs, ys, rs = _pebble_grow(xs, ys, W, H, r0=0.01, max_iter=1000, dt=0.0005)
    
    return np.concatenate([[W], xs, ys, rs])

def _polish(circles, W, H, time_budget=5.0):
    """Polish a valid packing by trying to grow radii slightly."""
    t0 = time.time()
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = circles[:, 2].copy()
    
    # Try to grow each circle slightly
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        for _ in range(50):
            # Compute minimum distance to boundary and neighbors
            min_d = min(xs[i] - rs[i], W - xs[i] - rs[i], 
                       ys[i] - rs[i], H - ys[i] - rs[i])
            for j in range(N):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = min(min_d, d - rs[i] - rs[j])
            
            if min_d < 1e-6:
                break
            rs[i] += min(min_d, 0.001)
    
    circles_out = np.column_stack([xs, ys, rs])
    if _is_valid(circles_out, W, H):
        return circles_out
    return circles

def construct():
    t0 = time.time()
    best_circles = None
    best_val = -np.inf
    best_W = 1.0
    best_H = 1.0
    
    # Initial W values to try (width of rectangle, H = 2 - W)
    W_starts = [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.9, 1.1, 0.75, 1.05, 0.95]
    
    for W0 in W_starts:
        if time.time() - t0 > DEADLINE:
            break
        
        H0 = 2.0 - W0
        
        # Generate initializations
        inits = []
        
        # Grids
        grid_configs = [(5, 5), (6, 4), (4, 6), (7, 4), (4, 7), (5, 4), (3, 7), (7, 3)]
        for nc, nr in grid_configs:
            if nc * nr >= N:
                inits.append(_grid_init(W0, H0, nc, nr))
        
        # Random
        for seed in range(5):
            inits.append(_random_init(W0, H0, seed=seed))
        
        # Pebble growth initializations
        for seed in range(3):
            inits.append(_pebble_init(W0, H0, seed=seed))
        
        for x0 in inits:
            if time.time() - t0 > DEADLINE:
                break
            
            # Multi-stage penalty optimization
            try:
                x_current = x0.copy()
                mu = 1.0
                
                # Stage 1: moderate penalty, find feasible region
                for stage in range(4):
                    if time.time() - t0 > DEADLINE:
                        break
                    
                    # Increase penalty each stage
                    mu = mu * 4.0  # 4, 16, 64, 256
                    
                    maxiter = 150 if stage < 2 else 250
                    res = minimize(
                        _objective, x_current,
                        args=(mu,),
                        method='L-BFGS-B',
                        jac=_objective_grad,
                        bounds=[(0.3, 1.7)] +  # W
                               [(0, W0+0.5)] * N +  # x positions
                               [(0, H0+0.5)] * N +  # y positions
                               [(0, 1.0)] * N,       # radii
                        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': 0}
                    )
                    
                    x_current = res.x
                    
                    # Check if we're close to feasible
                    circles, W_cur, H_cur, val = _extract_and_check(x_current)
                    if circles is not None:
                        if val > best_val:
                            best_val = val
                            best_circles = circles.copy()
                            best_W = W_cur
                            best_H = H_cur
                
                # Final polish with high penalty
                if time.time() - t0 < DEADLINE - 10:
                    mu_final = 500.0
                    res = minimize(
                        _objective, x_current,
                        args=(mu_final,),
                        method='L-BFGS-B',
                        jac=_objective_grad,
                        bounds=[(0.3, 1.7)] +
                               [(0, 2.0)] * N +
                               [(0, 2.0)] * N +
                               [(0, 1.0)] * N,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': 0}
                    )
                    x_current = res.x
                    circles, W_cur, H_cur, val = _extract_and_check(x_current)
                    if circles is not None:
                        if val > best_val:
                            best_val = val
                            best_circles = circles.copy()
                            best_W = W_cur
                            best_H = H_cur
                    
                    # Try to polish the result
                    if best_circles is not None and time.time() - t0 < DEADLINE - 5:
                        polished = _polish(best_circles, best_W, best_H, time_budget=3.0)
                        if _is_valid(polished, best_W, best_H):
                            val_pol = np.sum(polished[:, 2])
                            if val_pol > best_val:
                                best_val = val_pol
                                best_circles = polished.copy()
                
            except Exception:
                continue
    
    # Fallback: simple grid
    if best_circles is None:
        W = H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
