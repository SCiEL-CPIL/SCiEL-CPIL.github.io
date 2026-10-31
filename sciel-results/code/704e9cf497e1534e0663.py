import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def _validate(cx, cy, r, W, H, tol=1e-9):
    """Check if circles fit in rectangle [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_directed_relax(cx, cy, r, W, H, max_iter=500, dt=0.005):
    """
    Simple force-directed relaxation to fix overlaps and wall violations.
    Returns updated cx, cy. Radii are kept fixed.
    """
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Wall repulsion
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (r[i] - cx[i]) * 10
            if cy[i] - r[i] < 0:
                fy[i] += (r[i] - cy[i]) * 10
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10
            
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d_sq = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d_sq < min_d * min_d:
                    d = np.sqrt(d_sq)
                    if d < 1e-12:
                        dx = np.random.uniform(-0.01, 0.01)
                        dy = np.random.uniform(-0.01, 0.01)
                        d = np.sqrt(dx*dx + dy*dy)
                    overlap = min_d - d
                    fx[i] += (dx / d) * overlap * 5
                    fy[i] += (dy / d) * overlap * 5
                    fx[j] -= (dx / d) * overlap * 5
                    fy[j] -= (dy / d) * overlap * 5
        
        cx += dt * fx
        cy += dt * fy
        
        # Clamp to rectangle
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
    return cx, cy

def _max_radius_at(x, y, cx, cy, r, W, H):
    """
    Find the maximum radius of a circle centered at (x, y) that fits in [0, W] x [0, H]
    and does not overlap existing circles.
    """
    # Distance to walls
    r_max = min(x, W - x, y, H - y)
    
    # Distance to existing circles
    for i in range(len(cx)):
        dx = x - cx[i]
        dy = y - cy[i]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < r[i]:
            # Center is inside an existing circle, which shouldn't happen if we're careful, 
            # but if it does, no radius is possible.
            return 0.0
        # The new circle must be at least r[i] away from the center of circle i
        # dist >= r_new + r[i] => r_new <= dist - r[i]
        r_max = min(r_max, dist - r[i])
        
    return max(0.0, r_max)

def _sequential_packing(W, H, order, rng):
    """
    Pack circles sequentially in the given order.
    For each circle, try to find the best position and maximum radius.
    """
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    # Candidate positions: grid points
    # We can use a finer grid to find better spots
    grid_res = 40
    xs = np.linspace(0, W, grid_res)
    ys = np.linspace(0, H, grid_res)
    
    for idx in order:
        best_r = 0.0
        best_x = W / 2.0
        best_y = H / 2.0
        
        # Search on grid
        for x in xs:
            for y in ys:
                r_try = _max_radius_at(x, y, cx[:idx], cy[:idx], r[:idx], W, H)
                if r_try > best_r:
                    best_r = r_try
                    best_x = x
                    best_y = y
                    
        # If best_r is 0, we might be stuck. 
        # Try to relax a bit or just place a tiny circle? 
        # If we can't place it, the packing is invalid.
        # But with 21 circles in a perimeter 4 rect, we should be able to place them.
        
        # Refine position around best_x, best_y?
        # For speed, we just accept the grid point.
        
        cx[idx] = best_x
        cy[idx] = best_y
        r[idx] = best_r
        
    # After placing all, we might have some slack.
    # We can try to expand radii slightly if possible, or just return.
    # The sequential method tends to be conservative.
    
    # Force relax to fix any tiny numerical issues or overlaps caused by grid discretization
    # Note: force relax keeps radii fixed.
    # We need to ensure validity.
    if not _validate(cx, cy, r, W, H):
         # Try to fix with force relax
         cx, cy = _force_directed_relax(cx, cy, r, W, H, max_iter=100)
         if not _validate(cx, cy, r, W, H):
             return None
             
    return cx, cy, r

def _optimize_packing(W, H, seed=0, deadline=None):
    """
    Optimize packing for fixed W, H.
    Combines sequential packing initialization with local optimization.
    """
    rng = np.random.default_rng(seed)
    
    best_val = -np.inf
    best_x = None
    
    # Strategy 1: Sequential Packing with different orders
    # Try a few random orders
    for _ in range(3):
        if deadline and time.time() > deadline:
            break
        order = rng.permutation(N)
        res = _sequential_packing(W, H, order, rng)
        if res is not None:
            cx, cy, r = res
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_x = np.concatenate([cx, cy, r])
                
    # Strategy 2: Grid Initialization + L-BFGS-B
    # This is the original approach, but we can use the best sequential result as a start if available.
    
    layouts = [(5, 5), (4, 6), (3, 7), (7, 3), (6, 4)]
    
    for cols, rows in layouts:
        if cols * rows < N:
            continue
        if deadline and time.time() > deadline:
            break
            
        cell_w = W / cols
        cell_h = H / rows
        
        cx_init = np.zeros(N)
        cy_init = np.zeros(N)
        r_init = np.zeros(N)
        
        for k in range(N):
            i = k % cols
            j = k // cols
            cx_init[k] = (i + 0.5) * cell_w + rng.uniform(-0.1, 0.1) * cell_w
            cy_init[k] = (j + 0.5) * cell_h + rng.uniform(-0.1, 0.1) * cell_h
            r_init[k] = 0.4 * min(cell_w, cell_h)
            
        cx_rel, cy_rel = _force_directed_relax(cx_init, cy_init, r_init, W, H, iterations=100)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = cx_rel
        x0[1::3] = cy_rel
        x0[2::3] = r_init
        
        def _penalty_objective(x):
            cx = x[0::3]
            cy = x[1::3]
            r = x[2::3]
            r = np.maximum(r, 1e-6)
            
            penalty = 0.0
            
            for i in range(N):
                if cx[i] < r[i]:
                    penalty += (r[i] - cx[i]) ** 2
                if cx[i] + r[i] > W:
                    penalty += (cx[i] + r[i] - W) ** 2
                if cy[i] < r[i]:
                    penalty += (r[i] - cy[i]) ** 2
                if cy[i] + r[i] > H:
                    penalty += (cx[i] + r[i] - H) ** 2
                    
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
                        
            return -np.sum(r) + 100.0 * penalty

        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, min(W, H)/2))
            
        try:
            res = minimize(_penalty_objective, x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            
            cx_opt = res.x[0::3]
            cy_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            if _validate(cx_opt, cy_opt, r_opt, W, H):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_x = res.x
                    
        except Exception:
            pass
            
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_val = -np.inf
    best_result = None
    
    # We need W + H = 2.
    # Try a range of widths.
    widths = np.linspace(1.0, 1.8, 15)
    
    for idx, W in enumerate(widths):
        if time.time() > deadline - 5.0:
            break
            
        H = 2.0 - W
        
        # Try a few seeds for this aspect ratio
        seeds_to_try = [0, 1]
        if idx > 5:
            seeds_to_try = [0]
            
        for seed in seeds_to_try:
            if time.time() > deadline - 2.0:
                break
                
            x_opt, val = _optimize_packing(W, H, seed=seed, deadline=deadline - 2.0)
            
            if val > best_val:
                best_val = val
                best_result = x_opt
                
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx = best_result[0::3]
    cy = best_result[1::3]
    r = best_result[2::3]
    
    return np.column_stack([cx, cy, r])
# EVOLVE-BLOCK-END
