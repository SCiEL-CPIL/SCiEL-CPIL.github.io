"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: sequential greedy insertion (largest empty circle via distance transform)
+ per-circle convex sub-problem relaxation + final joint polish.
"""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _largest_empty_circle(others_xy, others_r, grid_res=256):
    """
    Find center of largest empty circle given fixed others + walls.
    Returns (x, y, max_radius) of the best placement.
    Uses distance transform on a fine grid.
    """
    # Build occupancy mask: 1 where a new circle of radius 0 would NOT fit
    # i.e., inside existing circles (shrunk) or outside [0,1]^2
    # We compute, for each grid point, the distance to the nearest "obstacle boundary"
    # Obstacles: the interior of each existing circle (distance to circle center - r)
    # and the outside of [0,1]^2.
    
    # For distance transform, we need a binary mask of "forbidden" region
    # A point is forbidden if it's inside any circle OR outside [0,1]^2
    # But we want the distance to the NEAREST point that is forbidden (or the boundary)
    
    # Actually, for a new circle of radius r_new at position p:
    #   p must satisfy: dist(p, c_i) >= r_i + r_new for all i
    #   and p must satisfy: r_new <= min(p_x, 1-p_x, p_y, 1-p_y)
    # So r_new(p) = min( min_i (dist(p, c_i) - r_i), min(p_x, 1-p_x, p_y, 1-p_y) )
    # We want to maximize r_new(p) over p in [0,1]^2.
    
    # Build a grid
    gx = np.linspace(0, 1, grid_res)
    gy = np.linspace(0, 1, grid_res)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    
    # Wall clearance at each grid point
    wall_clear = np.minimum(np.minimum(GX, 1.0 - GX), np.minimum(GY, 1.0 - GY))
    
    # Circle clearance at each grid point
    if len(others_xy) > 0:
        # For each grid point, compute min over circles of (dist - r_i)
        # Vectorize: for each circle, compute distance to all grid points
        circle_clear = np.full((grid_res, grid_res), np.inf)
        for k in range(len(others_xy)):
            cx, cy, cr = others_xy[k, 0], others_xy[k, 1], others_r[k]
            dist = np.sqrt((GX - cx)**2 + (GY - cy)**2)
            circle_clear = np.minimum(circle_clear, dist - cr)
    
    max_r_grid = np.minimum(wall_clear, circle_clear)
    
    # Find the max
    idx = np.unravel_index(np.argmax(max_r_grid), max_r_grid.shape)
    best_x = gx[idx[0]]
    best_y = gy[idx[1]]
    best_r = max_r_grid[idx]
    
    # Refine with a finer local grid around the best point
    if best_r > 0.001:
        step = 1.0 / grid_res
        fine_res = 32
        fx = np.linspace(max(0, best_x - 3*step), min(1, best_x + 3*step), fine_res)
        fy = np.linspace(max(0, best_y - 3*step), min(1, best_y + 3*step), fine_res)
        FGX, FGY = np.meshgrid(fx, fy, indexing='ij')
        fwall = np.minimum(np.minimum(FGX, 1.0 - FGX), np.minimum(FGY, 1.0 - FGY))
        if len(others_xy) > 0:
            fcirc = np.full((fine_res, fine_res), np.inf)
            for k in range(len(others_xy)):
                cx, cy, cr = others_xy[k, 0], others_xy[k, 1], others_r[k]
                fdist = np.sqrt((FGX - cx)**2 + (FGY - cy)**2)
                fcirc = np.minimum(fcirc, fdist - cr)
            fmax = np.minimum(fwall, fcirc)
        else:
            fmax = fwall
        fidx = np.unravel_index(np.argmax(fmax), fmax.shape)
        best_x = fx[fidx[0]]
        best_y = fy[fidx[1]]
        best_r = fmax[fidx]
    
    return best_x, best_y, max(0.0, best_r)


def _optimize_single_circle(x0, y0, r0, others_xy, others_r, deadline, max_iter=200):
    """
    Optimize a single circle's (x, y, r) given all others fixed.
    This is a small-scale problem (3 variables) - use L-BFGS-B.
    """
    n_others = len(others_xy)
    
    def obj(v):
        if time.time() > deadline:
            return 1e10
        x, y, r = v
        if r < 0:
            r = 0
        # Wall violations
        wall_min = min(x, 1.0 - x, y, 1.0 - y)
        wall_viol = max(0.0, r - wall_min)
        # Circle violations
        circ_viol = 0.0
        for k in range(n_others):
            dx = x - others_xy[k, 0]
            dy = y - others_xy[k, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = r + others_r[k] - dist
            if viol > 0:
                circ_viol += viol
        # Objective: maximize r, penalize violations
        return -r + 1000.0 * (wall_viol**2 + circ_viol**2)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3)
        x, y, r = v
        g = np.zeros(3)
        
        # d(-r)/dr = -1
        g[2] = -1.0
        
        # Wall penalty gradient
        wall_min = min(x, 1.0 - x, y, 1.0 - y)
        wall_viol = max(0.0, r - wall_min)
        if wall_viol > 0:
            # d(wall_viol^2)/d(var) = 2*wall_viol * d(wall_viol)/d(var)
            pw = 2.0 * 1000.0 * wall_viol
            # d(wall_viol)/dr = 1
            g[2] += pw
            # d(wall_viol)/dx depends on which wall is binding
            if x <= 1.0 - x and x <= y and x <= 1.0 - y:
                g[0] -= pw  # wall_min = x
            elif 1.0 - x <= x and 1.0 - x <= y and 1.0 - x <= 1.0 - y:
                g[0] += pw  # wall_min = 1-x
            elif y <= x and y <= 1.0 - x and y <= 1.0 - y:
                g[1] -= pw  # wall_min = y
            else:
                g[1] += pw  # wall_min = 1-y
        
        # Circle penalty gradient
        for k in range(n_others):
            dx = x - others_xy[k, 0]
            dy = y - others_xy[k, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = r + others_r[k] - dist
            if viol > 0:
                pc = 2.0 * 1000.0 * viol
                # d(viol)/dx = dx/dist, d(viol)/dy = dy/dist, d(viol)/dr = 1
                g[0] += pc * dx / dist
                g[1] += pc * dy / dist
                g[2] += pc
        
        return g
    
    v0 = np.array([x0, y0, r0])
    bounds = [(0.0001, 0.9999), (0.0001, 0.9999), (0.0001, 0.4999)]
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        v_opt = result.x
        # Ensure validity
        x, y, r = v_opt
        # Clamp r to feasible
        wall_min = min(x, 1.0 - x, y, 1.0 - y)
        r = min(r, wall_min)
        for k in range(n_others):
            dx = x - others_xy[k, 0]
            dy = y - others_xy[k, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r = min(r, max(0, dist - others_r[k]))
        r = max(r, 0.0001)
        return x, y, r
    except Exception:
        return x0, y0, r0


def _sequential_insertion(deadline, grid_res=256, polish_each=True):
    """
    Sequentially insert N circles, each at the largest empty circle position,
    optionally polishing each new circle against the fixed existing set.
    """
    others_xy = np.zeros((0, 2))
    others_r = np.zeros(0)
    
    for i in range(N):
        if time.time() > deadline - 5:
            break
        
        # Find largest empty circle
        x, y, r = _largest_empty_circle(others_xy, others_r, grid_res=grid_res)
        
        # Polish this circle
        if polish_each and r > 0.001:
            x, y, r = _optimize_single_circle(x, y, r, others_xy, others_r, deadline)
        
        others_xy = np.vstack([others_xy, [[x, y]]])
        others_r = np.append(others_r, r)
    
    if len(others_xy) < N:
        # Fill remaining with tiny circles
        while len(others_xy) < N:
            x, y, r = _largest_empty_circle(others_xy, others_r, grid_res=128)
            if r < 0.001:
                x, y, r = 0.5, 0.5, 0.001
            others_xy = np.vstack([others_xy, [[x, y]]])
            others_r = np.append(others_r, r)
    
    return others_xy, others_r


def _joint_polish(xy, r, deadline, beta=100.0, pw=200.0, max_iter=1000):
    """
    Final joint optimization of all circles using smooth objective + L-BFGS-B.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))
    
    # Precompute pair indices for speed
    pair_idx = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        x = v[0:n]
        y = v[n:2*n]
        rr = v[2*n:3*n]
        
        # Wall violations (smooth)
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)
        
        # Pairwise violations (smooth)
        total_pen = 0.0
        for i, j in pair_idx:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = rr[i] + rr[j] - dist
            total_pen += np.logaddexp(0.0, beta * viol) / beta
        
        return -np.sum(rr) + pw * (wall_pen + total_pen)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        x = v[0:n]
        y = v[n:2*n]
        rr = v[2*n:3*n]
        
        g = np.zeros(3 * n)
        
        # Wall gradients
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        sig = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        
        for i in range(n):
            s = sig[i]
            if s < 1e-12:
                continue
            dists = np.array([x[i], 1.0-x[i], y[i], 1.0-y[i]])
            mi = int(np.argmin(dists))
            f = pw * s
            g[2*n+i] += f
            if mi == 0:
                g[i] -= f
            elif mi == 1:
                g[i] += f
            elif mi == 2:
                g[n+i] -= f
            else:
                g[n+i] += f
        
        # Pairwise gradients
        for i, j in pair_idx:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 1e-20:
                continue
            dist = np.sqrt(dist_sq)
            viol = rr[i] + rr[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s < 1e-12:
                continue
            f = pw * s
            g[i] += f * dx / dist
            g[n+i] += f * dy / dist
            g[j] -= f * dx / dist
            g[n+j] -= f * dy / dist
            g[2*n+i] += f
            g[2*n+j] += f
        
        g[2*n:3*n] -= 1.0
        return g
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        return new_xy, new_r
    except Exception:
        return xy, r


def _local_search_refine(xy, r, deadline, grid_res=128, max_sweeps=5):
    """
    Coordinate descent: for each circle, find best position+radius given others fixed.
    Uses the largest empty circle computation for each circle.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 3:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]
            
            # Find best position for circle i
            x, y, r_new = _largest_empty_circle(others_xy, others_r, grid_res=grid_res)
            
            if r_new > r[i] + 1e-10:
                xy[i, 0] = x
                xy[i, 1] = y
                r[i] = r_new
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Phase 1: Sequential insertion with per-circle polish
    if time.time() < deadline - 30:
        xy, r = _sequential_insertion(deadline, grid_res=256, polish_each=True)
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
    
    # Phase 2: Local search refinement (coordinate descent)
    if best_valid and best_xy is not None and time.time() < deadline - 20:
        ls_xy, ls_r = _local_search_refine(best_xy.copy(), best_r.copy(), deadline, grid_res=128, max_sweeps=5)
        total, valid = _eval_packing(ls_xy, ls_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = ls_xy
            best_r = ls_r
    
    # Phase 3: Joint polish (multi-stage beta)
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        for beta, pw, max_it in [(50.0, 100.0, 500), (100.0, 200.0, 800), (200.0, 500.0, 1000)]:
            if time.time() > deadline - 5:
                break
            pol_xy, pol_r = _joint_polish(best_xy, best_r, deadline, beta=beta, pw=pw, max_iter=max_it)
            total, valid = _eval_packing(pol_xy, pol_r)
            if valid and total > best_total + 1e-12:
                best_total = total
                best_valid = True
                best_xy = pol_xy
                best_r = pol_r
    
    # Phase 4: Another local search pass
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        ls2_xy, ls2_r = _local_search_refine(best_xy.copy(), best_r.copy(), deadline, grid_res=100, max_sweeps=3)
        total, valid = _eval_packing(ls2_xy, ls2_r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = ls2_xy
            best_r = ls2_r
    
    # Phase 5: Final joint polish
    if best_valid and best_xy is not None and time.time() < deadline - 2:
        pol_xy, pol_r = _joint_polish(best_xy, best_r, deadline, beta=200.0, pw=500.0, max_iter=1500)
        total, valid = _eval_packing(pol_xy, pol_r)
        if valid and total > best_total + 1e-12:
            best_total = total
            best_valid = True
            best_xy = pol_xy
            best_r = pol_r
    
    # Fallback
    if best_xy is None or not best_valid:
        # Simple grid fallback
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
