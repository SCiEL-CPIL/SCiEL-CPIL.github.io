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

def _objective(params, start):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    # Vectorized overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    # Boundary penalties
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-8))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 50.0
    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start):
    """Tighter penalty objective for final polishing with L-BFGS-B."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    # Vectorized overlap penalty with tighter smoothing
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-12))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    # Boundary penalties with tighter smoothing
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-12))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    mu = 200.0
    return obj + mu * (overlap_pen + bound_pen)

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

def _physics_optimize(x, y, r, start, max_time=10.0):
    """
    Physics-based optimization: treat circles as particles with repulsive forces
    when overlapping and attractive forces when close. Also apply pressure to grow radii.
    Uses velocity Verlet integration with damping.
    """
    x = x.astype(float).copy()
    y = y.astype(float).copy()
    r = r.astype(float).copy()
    vx = np.zeros(N)
    vy = np.zeros(N)
    vr = np.zeros(N)
    
    dt = 0.002
    damping = 0.92
    repulse_k = 50.0    # repulsion strength for overlaps
    attract_k = 2.0     # attraction for near-contact (helps find dense packing)
    grow_k = 1.5        # pressure to grow radii
    wall_k = 80.0       # wall repulsion strength
    
    start_phys = time.time()
    max_iters = int(max_time / dt)
    
    for it in range(max_iters):
        if time.time() - start_phys > max_time or _elapsed(start) > TIME_LIMIT - 5:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Pairwise forces (vectorized)
        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat*dx_mat + dy_mat*dy_mat + 1e-14)
        need_mat = r[:, None] + r[None, :]
        
        # Overlap mask (upper triangle only)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_mask = (dist_mat < need_mat) & mask
        near_mask = (dist_mat >= need_mat) & (dist_mat < need_mat * 1.05) & mask
        
        # Process overlaps
        for i, j in zip(*np.where(overlap_mask)):
            d = dist_mat[i, j]
            need = need_mat[i, j]
            overlap = need - d
            if d < 1e-12:
                ux, uy = 1.0, 0.0
            else:
                ux, uy = dx_mat[i, j] / d, dy_mat[i, j] / d
            f = repulse_k * overlap
            fx[i] += f * ux
            fy[i] += f * uy
            fx[j] -= f * ux
            fy[j] -= f * uy
            fr[i] -= 0.3 * f
            fr[j] -= 0.3 * f
        
        # Process near-contact (attraction)
        for i, j in zip(*np.where(near_mask)):
            d = dist_mat[i, j]
            gap = d - need_mat[i, j]
            if d < 1e-12:
                continue
            ux, uy = dx_mat[i, j] / d, dy_mat[i, j] / d
            f = attract_k * gap
            fx[i] -= f * ux
            fy[i] -= f * uy
            fx[j] += f * ux
            fy[j] += f * uy
        
        # Wall forces: push circles away from walls
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += wall_k * pen
            # Right wall
            if x[i] > 1.0 - r[i]:
                pen = x[i] - (1.0 - r[i])
                fx[i] -= wall_k * pen
            # Bottom wall
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += wall_k * pen
            # Top wall
            if y[i] > 1.0 - r[i]:
                pen = y[i] - (1.0 - r[i])
                fy[i] -= wall_k * pen
        
        # Growth force: try to grow each radius
        for i in range(N):
            # Max possible radius given position
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] < max_r * 0.95:
                fr[i] += grow_k * (max_r * 0.95 - r[i])
        
        # Velocity Verlet update
        vx = vx * damping + fx * dt
        vy = vy * damping + fy * dt
        vr = vr * damping + fr * dt
        
        x = x + vx * dt
        y = y + vy * dt
        r = r + vr * dt
        
        # Clamp
        r = np.maximum(r, 0.0)
        r = np.minimum(r, 0.5)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        # Zero velocities at walls
        for i in range(N):
            if x[i] <= r[i] + 1e-9 or x[i] >= 1.0 - r[i] - 1e-9:
                vx[i] = 0
            if y[i] <= r[i] + 1e-9 or y[i] >= 1.0 - r[i] - 1e-9:
                vy[i] = 0
        
        # If no overlaps and radii not growing, we're at a local optimum
        if it > 100:
            if not np.any(overlap_mask):
                # Check if radii are still growing
                if np.max(np.abs(vr)) < 1e-6:
                    break
    
    return x, y, r

def _tight_polish(x, y, r, start):
    """
    High-precision final polish using alternating:
    1. Fix radii, optimize positions with L-BFGS-B (tighter)
    2. Fix positions, optimize radii with SLSQP
    3. Repeat
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    best_score = _sum_r(r)
    
    for outer in range(5):
        if _elapsed(start) > TIME_LIMIT - 3:
            break
        
        # Step 1: Optimize positions with fixed radii
        try:
            def pos_obj(params):
                if _elapsed(start) > TIME_LIMIT:
                    return 1e10
                px = params[:N]
                py = params[N:2*N]
                obj = 0.0
                # Overlap penalty
                dx = px[:, None] - px[None, :]
                dy = py[:, None] - py[None, :]
                dist = np.sqrt(dx*dx + dy*dy + 1e-14)
                need = r[:, None] + r[None, :]
                diff = need - dist
                smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-14))
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                obj += 100.0 * np.sum(smooth_max[mask])
                # Boundary
                for i in range(N):
                    if px[i] < r[i]:
                        obj += 100.0 * (r[i] - px[i])**2
                    if px[i] > 1.0 - r[i]:
                        obj += 100.0 * (px[i] - (1.0 - r[i]))**2
                    if py[i] < r[i]:
                        obj += 100.0 * (r[i] - py[i])**2
                    if py[i] > 1.0 - r[i]:
                        obj += 100.0 * (py[i] - (1.0 - r[i]))**2
                return obj
            
            p0 = np.concatenate([x, y])
            bounds = [(r[i], 1.0 - r[i]) for i in range(N)] * 2
            res = minimize(pos_obj, p0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
            x_new = res.x[:N]
            y_new = res.x[N:2*N]
            if _is_valid(x_new, y_new, r):
                x, y = x_new, y_new
        except Exception:
            pass
        
        # Step 2: Optimize radii with fixed positions
        try:
            def rad_obj(params):
                if _elapsed(start) > TIME_LIMIT:
                    return 1e10
                pr = np.maximum(params, 0.0)
                obj = -np.sum(pr)
                # Overlap penalty
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx*dx + dy*dy + 1e-14)
                need = pr[:, None] + pr[None, :]
                diff = need - dist
                smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-14))
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                obj += 200.0 * np.sum(smooth_max[mask])
                # Boundary
                for i in range(N):
                    if pr[i] > x[i]:
                        obj += 200.0 * (pr[i] - x[i])**2
                    if pr[i] > 1.0 - x[i]:
                        obj += 200.0 * (pr[i] - (1.0 - x[i]))**2
                    if pr[i] > y[i]:
                        obj += 200.0 * (pr[i] - y[i])**2
                    if pr[i] > 1.0 - y[i]:
                        obj += 200.0 * (pr[i] - (1.0 - y[i]))**2
                return obj
            
            p0 = r.copy()
            bounds_r = [(0.0, min(x[i], 1.0-x[i], y[i], 1.0-y[i])) for i in range(N)]
            res = minimize(rad_obj, p0, method='L-BFGS-B', bounds=bounds_r,
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
            r_new = np.maximum(res.x, 0.0)
            if _is_valid(x, y, r_new):
                s = _sum_r(r_new)
                if s > best_score:
                    best_score = s
                    r = r_new
        except Exception:
            pass
    
    return x, y, r

def _make_valid(x, y, r):
    """Guarantee validity: push apart, then shrink any residual overlap/bounds."""
    x, y = _push_apart(x, y, r)
    # clamp
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    # residual fixes
    for _ in range(300):
        ok = True
        for i in range(N):
            mx = min(x[i], 1.0 - x[i])
            my = min(y[i], 1.0 - y[i])
            mr = min(mx, my)
            if r[i] > mr:
                r[i] = max(0.0, mr * 0.999)
                ok = False
        if not ok:
            continue
        # Vectorized overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need
        if np.any(overlaps):
            # Find first overlap
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            f = 0.995
            r[i] *= f
            r[j] *= f
            ok = False
        if ok:
            break
    return x, y, r

def _hex_init():
    # Hexagonal packing for 32 circles, then scale to fit square
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
    # radius = half min neighbor distance, capped by boundary
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
    
    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            res = minimize(
                _objective, params, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Physics-based optimization
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            px, py, pr = _physics_optimize(best[0], best[1], best[2], start, max_time=8.0)
            px, py, pr = _make_valid(px, py, pr)
            if _is_valid(px, py, pr):
                s = _sum_r(pr)
                if s > best_score:
                    best_score = s
                    best = (px, py, pr)
        except Exception:
            pass
    
    # Powell polish on best
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # L-BFGS-B with tight penalty for final high-precision polish
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            # Bounds for L-BFGS-B: x in [r, 1-r], y in [r, 1-r], r in [0, 0.5]
            # Since r varies, use loose bounds and rely on penalty
            bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
            res = minimize(
                _objective_tight, p0, args=(start,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Tight polish: alternate position and radius optimization
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            tx, ty, tr = _tight_polish(best[0], best[1], best[2], start)
            tx, ty, tr = _make_valid(tx, ty, tr)
            if _is_valid(tx, ty, tr):
                s = _sum_r(tr)
                if s > best_score:
                    best_score = s
                    best = (tx, ty, tr)
        except Exception:
            pass
            
    return best

def _voronoi_init():
    """
    Generate a packing based on Voronoi cells of a random point set,
    then optimize the centers/radii.
    """
    rng = np.random.RandomState(42)
    # Generate points in a grid-like random pattern
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    
    # Compute Voronoi diagram to find inradii
    try:
        tri = Delaunay(np.column_stack((x, y)))
        # For each point, find its Voronoi cell inradius
        # This is complex, so we approximate by distance to nearest neighbors
        r = np.zeros(N)
        for i in range(N):
            # Distance to nearest neighbors
            dists = np.sqrt((x[i]-x)**2 + (y[i]-y)**2)
            dists[i] = np.inf
            # Take the 3 nearest neighbors to get a robust radius estimate
            nearest = np.argsort(dists)[:3]
            min_dist = np.min(dists[nearest])
            r[i] = 0.45 * min_dist
            # Cap by boundary
            r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    except Exception:
        r = np.full(N, 0.05)
        
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
        if _elapsed(start) > TIME_LIMIT - 10:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final global polish on best
    if _elapsed(start) < TIME_LIMIT - 8:
        res = _optimize(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    # Final tight polish
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            tx, ty, tr = _tight_polish(best[0], best[1], best[2], start)
            tx, ty, tr = _make_valid(tx, ty, tr)
            if _is_valid(tx, ty, tr):
                s = _sum_r(tr)
                if s > best_score:
                    best = (tx, ty, tr)
        except Exception:
            pass

    x, y, r = best
    # ensure final validity
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
