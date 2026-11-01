import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32
TIME_LIMIT = 135.0  # evaluator kills at 150s; keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    # bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # pairwise disjoint interiors
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

def _penalty(params, start):
    """
    Objective function for L-BFGS-B.
    Minimize: -sum(r) + penalty(overlaps) + penalty(bounds)
    """
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
        
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Ensure radii are non-negative for calculation stability, though L-BFGS-B handles bounds
    r_eff = np.maximum(r, 0.0)
    
    obj = -np.sum(r_eff)
    
    # Vectorized overlap penalty (quadratic)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    min_dist = r_eff[:, None] + r_eff[None, :]
    diff = min_dist - dist
    # Smooth max approximation for differentiability, or just max(0, diff)^2
    # Using max(0, diff)^2 is standard for L-BFGS-B
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(np.maximum(0, diff[mask])**2)
    
    # Boundary penalties (quadratic)
    dneg_x = r_eff - x
    dpos_x = x - (1.0 - r_eff)
    dneg_y = r_eff - y
    dpos_y = y - (1.0 - r_eff)
    
    bound_pen = np.sum(np.maximum(0, dneg_x)**2) + \
                np.sum(np.maximum(0, dpos_x)**2) + \
                np.sum(np.maximum(0, dneg_y)**2) + \
                np.sum(np.maximum(0, dpos_y)**2)
    
    # Weights: High enough to enforce constraints, low enough to allow radius growth
    # If weights are too high, the optimizer just minimizes violations and ignores radius growth
    # If weights are too low, it grows radii into overlap.
    # Adaptive weighting could be used, but fixed high weight is safer for validity.
    mu_overlap = 100.0
    mu_bound = 100.0
    
    return obj + mu_overlap * overlap_pen + mu_bound * bound_pen

def _physics_relax(x, y, r, iters=1000, start=None):
    """
    Simple physics-based relaxation to spread circles.
    Uses repulsive forces to increase minimum distances.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(iters):
        if start is not None and _elapsed(start) > TIME_LIMIT - 5:
            break
            
        moved = False
        # Calculate forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                if d < 1e-6:
                    dx = (i - j) * 1e-6
                    dy = 1e-6
                    d = np.hypot(dx, dy)
                
                # Desired distance is sum of radii + small gap
                # We want to push them apart if they are too close
                # Force is proportional to overlap or inverse square
                # Let's use a strong repulsion if d < r_i + r_j
                # And a weak attraction to center? No, just repulsion and boundary repulsion.
                
                if d < r[i] + r[j]:
                    # Overlap: push apart strongly
                    f = 1.0 / (d * d + 1e-6)
                    fx[i] += f * (dx / d)
                    fy[i] += f * (dy / d)
                    fx[j] -= f * (dx / d)
                    fy[j] -= f * (dy / d)
                    moved = True
                else:
                    # No overlap: slight repulsion to maximize space?
                    # Actually, we want to maximize radii.
                    # If we move them further apart, we can increase radii.
                    # So apply a weak repulsive force always?
                    # This might cause them to fly to corners.
                    # Better: only push if they are "tight".
                    pass
        
        # Boundary repulsion
        for i in range(N):
            # Left
            if x[i] < r[i]:
                fx[i] += 1.0 / (max(r[i] - x[i], 1e-6)**2)
                moved = True
            # Right
            if x[i] > 1.0 - r[i]:
                fx[i] -= 1.0 / (max(x[i] - (1.0 - r[i]), 1e-6)**2)
                moved = True
            # Bottom
            if y[i] < r[i]:
                fy[i] += 1.0 / (max(r[i] - y[i], 1e-6)**2)
                moved = True
            # Top
            if y[i] > 1.0 - r[i]:
                fy[i] -= 1.0 / (max(y[i] - (1.0 - r[i]), 1e-6)**2)
                moved = True
            
            # Gentle pull to center to prevent drifting to edges?
            # fx[i] -= 0.01 * (x[i] - 0.5)
            # fy[i] -= 0.01 * (y[i] - 0.5)
        
        # Update positions with damping
        step_size = 0.01
        x += step_size * fx
        y += step_size * fy
        
        # Clamp to valid range (strictly inside)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        # Update radii to max possible for current positions
        for i in range(N):
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            for j in range(N):
                if i == j: continue
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                max_r = min(max_r, d - r[j])
            r[i] = max(0.0, max_r)
            
        if not moved:
            break
            
    return x, y, r

def _make_valid(x, y, r):
    """Guarantee validity and maximize radii for given centers."""
    # 1. Resolve overlaps and bounds iteratively
    for _ in range(10):
        x, y = _push_apart(x, y, r, iters=100)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
    # 2. Ensure current r is valid (shrink if necessary)
    for _ in range(50):
        ok = True
        # Bounds
        for i in range(N):
            mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > mr:
                r[i] = max(0.0, mr * 0.999)
                ok = False
        # Overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need - 1e-9
        if np.any(overlaps):
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            f = 0.99
            r[i] *= f
            r[j] *= f
            ok = False
        if ok:
            break
            
    # 3. Expand radii to maximal feasible size
    # Iterative relaxation: r_i = min(bound_i, min_j (dist_ij - r_j))
    for _ in range(100):
        r_max_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        new_r = np.zeros(N)
        for i in range(N):
            max_r = r_max_bound[i]
            for j in range(N):
                if i == j:
                    continue
                limit = dist[i, j] - r[j]
                if limit < max_r:
                    max_r = limit
            new_r[i] = max(0.0, max_r)
            
        if np.allclose(new_r, r, rtol=1e-6, atol=1e-8):
            r = new_r
            break
        r = new_r
        
    # 4. Final safety shrink for numerical precision
    for _ in range(50):
        ok = True
        for i in range(N):
            mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > mr + 1e-9:
                r[i] = mr
                ok = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need - 1e-9
        if np.any(overlaps):
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            f = 0.99
            r[i] *= f
            r[j] *= f
            ok = False
        if ok:
            break
            
    return x, y, r

def _push_apart(x, y, r, iters=500):
    """Move centers to resolve overlaps; never shrink radii. Keep in bounds."""
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        # Vectorized check for overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need
        
        if not np.any(overlaps):
            break
            
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / d, dy / d
                    gap = need_ij - d
                    # Move each half the gap along the separating axis
                    step = 0.5 * gap
                    x[i] += step * ux
                    y[i] += step * uy
                    x[j] -= step * ux
                    y[j] -= step * uy
                    moved = True
        # Clamp back into square (respecting radius)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if not moved:
            break
    return x, y

def _hex_init():
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.07)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Bounds for L-BFGS-B
    # x, y in [0, 1], r in [0, 0.5]
    # But we must respect r <= x <= 1-r etc.
    # L-BFGS-B handles box constraints. We will use loose box constraints and rely on penalty.
    # However, tighter bounds help.
    # x_min = r, x_max = 1-r. This is coupled.
    # Standard approach: box constraints on x,y,r and penalty for violations.
    bounds = [ (0, 1) ] * (2*N) + [ (0, 0.5) ] * N
    
    # 1. L-BFGS-B Optimization
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = minimize(
                _penalty, params, args=(start,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
            
    # 2. Physics Relaxation
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            vx, vy, vr = _physics_relax(best[0], best[1], best[2], iters=200, start=start)
            vx, vy, vr = _make_valid(vx, vy, vr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
            
    # 3. Local Perturbation Search (Escape local minima)
    if _elapsed(start) < TIME_LIMIT - 10:
        rng = np.random.RandomState(42)
        for _ in range(20):
            if _elapsed(start) > TIME_LIMIT - 10:
                break
            # Perturb centers slightly
            px = best[0].copy()
            py = best[1].copy()
            pr = best[2].copy()
            
            # Random small move
            for i in range(N):
                px[i] += rng.uniform(-0.05, 0.05)
                py[i] += rng.uniform(-0.05, 0.05)
            
            # Make valid (push apart, expand radii)
            vx, vy, vr = _make_valid(px, py, pr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
                    # Re-optimize from this new best
                    if _elapsed(start) < TIME_LIMIT - 5:
                         try:
                            p0 = np.concatenate([best[0], best[1], best[2]])
                            res = minimize(
                                _penalty, p0, args=(start,),
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                            )
                            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
                            if _is_valid(vx, vy, vr):
                                s = _sum_r(vr)
                                if s > best_score:
                                    best_score = s
                                    best = (vx, vy, vr)
                         except Exception:
                             pass

    return best

def _voronoi_init():
    rng = np.random.RandomState(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.zeros(N)
    for i in range(N):
        dists = np.sqrt((x[i]-x)**2 + (y[i]-y)**2)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:3]
        min_dist = np.min(dists[nearest])
        r[i] = 0.45 * min_dist
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    return x, y, r

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_rect_init(9, 4))
    inits.append(_rect_init(4, 9))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_voronoi_init())
    
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 15:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final global polish on best
    if _elapsed(start) < TIME_LIMIT - 5:
        res = _optimize(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    x, y, r = best
    # ensure final validity
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
