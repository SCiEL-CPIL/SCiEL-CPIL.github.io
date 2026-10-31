"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
TIME_LIMIT = 135  # seconds, leave margin for 150s kill

def _elapsed():
    return time.time() - _start_time

def _validate(pts, w, h):
    """Check if circles are valid (within bounds, no overlaps)."""
    n = len(pts)
    for i in range(n):
        x, y, r = pts[i]
        if x < r - 1e-7 or x > w - r + 1e-7 or y < r - 1e-7 or y > h - r + 1e-7:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < pts[i, 2] + pts[j, 2] - 1e-7:
                return False
    return True

def _sum_radii(pts):
    return np.sum(pts[:, 2])

def _objective(params, w, h):
    """
    Minimize: -sum(r) + penalty for overlaps and boundary violations.
    params: flat array [x1,y1,r1, x2,y2,r2, ...]
    """
    n = len(params) // 3
    pts = params.reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    obj = -np.sum(rs)
    
    C = 50.0
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                overlap_pen += (min_dist - dist) ** 2
    
    boundary_pen = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            boundary_pen += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            boundary_pen += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            boundary_pen += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            boundary_pen += (ys[i] - (h - rs[i])) ** 2
    
    return obj + C * (overlap_pen + boundary_pen)

def _init_hexagonal(w, h, n_circles, seed=0):
    """Initialize circles in a hexagonal packing pattern."""
    rng = np.random.RandomState(seed)
    
    # Estimate radius from area
    area = w * h
    avg_r = np.sqrt(area / (n_circles * np.pi))
    
    # Hexagonal grid parameters
    # Row spacing: sqrt(3)/2 * 2r = sqrt(3) * r
    # Col spacing: 2r
    # Offset alternating rows by r
    
    circles = []
    r_est = avg_r * 0.95
    
    row_spacing = np.sqrt(3) * r_est
    col_spacing = 2 * r_est
    
    # Start from top-left, go down and right
    row = 0
    while len(circles) < n_circles:
        offset = 0 if row % 2 == 0 else r_est
        col = 0
        while len(circles) < n_circles:
            x = r_est + offset + col * col_spacing
            y = r_est + row * row_spacing
            
            if x + r_est > w or y + r_est > h:
                break
            
            # Add small jitter
            x += (rng.random() - 0.5) * 0.05 * r_est
            y += (rng.random() - 0.5) * 0.05 * r_est
            
            circles.append([x, y, r_est])
            col += 1
        row += 1
    
    return np.array(circles[:n_circles])

def _init_grid(w, h, n_circles, rows, cols, seed=0):
    """Initialize circles in a grid pattern."""
    rng = np.random.RandomState(seed)
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = min(cell_w, cell_h) * 0.45
            x += (rng.random() - 0.5) * 0.1 * cell_w
            y += (rng.random() - 0.5) * 0.1 * cell_h
            circles.append([x, y, r])
    return np.array(circles)

def _optimize_from_init(pts_init, w, h, seed=0):
    """Run L-BFGS-B optimization from a given initialization."""
    params = pts_init.flatten()
    
    bounds = []
    for i in range(21):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0.005, min(w, h) * 0.45))
    
    try:
        result = minimize(_objective, params, args=(w, h),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12})
    except Exception:
        return None, -np.inf
    
    if _elapsed() > TIME_LIMIT:
        return None, -np.inf
    
    pts = result.x.reshape(21, 3)
    
    if not _validate(pts, w, h):
        # Try to repair
        pts = _repair(pts, w, h)
        if not _validate(pts, w, h):
            return None, -np.inf
    
    val = _sum_radii(pts)
    return pts, val

def _repair(pts, w, h):
    """Repair invalid packing by reducing radii and adjusting positions."""
    pts = pts.copy()
    for _ in range(20):
        # Clamp to bounds
        for i in range(21):
            x, y, r = pts[i]
            r = min(r, x, w - x, y, h - y)
            r = max(r, 0.005)
            pts[i, 2] = r
        
        # Fix overlaps
        overlap_found = False
        for i in range(21):
            for j in range(i + 1, 21):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pts[i, 2] + pts[j, 2]
                if dist < min_dist:
                    reduce = (min_dist - dist) / 2 + 0.002
                    pts[i, 2] = max(pts[i, 2] - reduce, 0.005)
                    pts[j, 2] = max(pts[j, 2] - reduce, 0.005)
                    overlap_found = True
        if not overlap_found:
            break
    
    return pts

def _simulated_annealing(pts, w, h, n_iter=500, seed=0):
    """
    Simulated annealing on discrete grid positions.
    Circles are moved between grid cells; radii are recalculated.
    """
    rng = np.random.RandomState(seed)
    
    # Create a grid of possible positions
    grid_res = 40
    xs_grid = np.linspace(0.05, w - 0.05, grid_res)
    ys_grid = np.linspace(0.05, h - 0.05, grid_res)
    
    # Current state
    current_pts = pts.copy()
    current_val = _sum_radii(current_pts)
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    best_pts = current_pts.copy()
    best_val = current_val
    
    for it in range(n_iter):
        if _elapsed() > TIME_LIMIT:
            break
        
        T = T_start * (cooling ** it)
        
        # Pick a random circle to move
        i = rng.randint(21)
        
        # Try a random new position
        new_x = rng.uniform(0.05, w - 0.05)
        new_y = rng.uniform(0.05, h - 0.05)
        
        # Recalculate radius for this circle
        # Radius is limited by distance to boundaries and neighbors
        r_new = min(new_x, w - new_x, new_y, h - new_y)
        for j in range(21):
            if j == i:
                continue
            dx = new_x - current_pts[j, 0]
            dy = new_y - current_pts[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            r_new = min(r_new, (dist - current_pts[j, 2]) * 0.95)
        r_new = max(r_new, 0.005)
        
        # Store old position
        old_x, old_y, old_r = current_pts[i]
        current_pts[i] = [new_x, new_y, r_new]
        
        # Recalculate all radii (since moving one circle affects others)
        for k in range(21):
            r_k = min(current_pts[k, 0], w - current_pts[k, 0],
                     current_pts[k, 1], h - current_pts[k, 1])
            for l in range(21):
                if l == k:
                    continue
                dx = current_pts[k, 0] - current_pts[l, 0]
                dy = current_pts[k, 1] - current_pts[l, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                r_k = min(r_k, (dist - current_pts[l, 2]) * 0.95)
            current_pts[k, 2] = max(r_k, 0.005)
        
        new_val = _sum_radii(current_pts)
        
        # Accept or reject
        if new_val > current_val or rng.random() < np.exp((new_val - current_val) / T):
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_pts = current_pts.copy()
        else:
            # Reject: restore
            current_pts[i] = [old_x, old_y, old_r]
            # Restore all radii
            for k in range(21):
                r_k = min(current_pts[k, 0], w - current_pts[k, 0],
                         current_pts[k, 1], h - current_pts[k, 1])
                for l in range(21):
                    if l == k:
                        continue
                    dx = current_pts[k, 0] - current_pts[l, 0]
                    dy = current_pts[k, 1] - current_pts[l, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    r_k = min(r_k, (dist - current_pts[l, 2]) * 0.95)
                current_pts[k, 2] = max(r_k, 0.005)
    
    return best_pts, best_val

def _force_directed(pts, w, h, n_iter=200):
    """
    Force-directed relaxation: push overlapping circles apart, pull toward center.
    """
    pts = pts.copy()
    dt = 0.05
    
    for it in range(n_iter):
        if _elapsed() > TIME_LIMIT:
            break
        
        forces = np.zeros_like(pts)
        
        for i in range(21):
            # Boundary forces
            if pts[i, 0] < pts[i, 2]:
                forces[i, 0] += (pts[i, 2] - pts[i, 0]) * 10
            if pts[i, 0] > w - pts[i, 2]:
                forces[i, 0] -= (pts[i, 0] - (w - pts[i, 2])) * 10
            if pts[i, 1] < pts[i, 2]:
                forces[i, 1] += (pts[i, 2] - pts[i, 1]) * 10
            if pts[i, 1] > h - pts[i, 2]:
                forces[i, 1] -= (pts[i, 1] - (h - pts[i, 2])) * 10
            
            # Overlap forces
            for j in range(i + 1, 21):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = pts[i, 2] + pts[j, 2]
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * 5
                    fy = (dy / dist) * overlap * 5
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Update positions
        for i in range(21):
            pts[i, 0] += forces[i, 0] * dt
            pts[i, 1] += forces[i, 1] * dt
            
            # Clamp to bounds
            pts[i, 0] = max(pts[i, 2], min(w - pts[i, 2], pts[i, 0]))
            pts[i, 1] = max(pts[i, 2], min(h - pts[i, 2], pts[i, 1]))
        
        # Recalculate radii
        for i in range(21):
            r = min(pts[i, 0], w - pts[i, 0], pts[i, 1], h - pts[i, 1])
            for j in range(21):
                if j == i:
                    continue
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                r = min(r, (dist - pts[j, 2]) * 0.98)
            pts[i, 2] = max(r, 0.005)
    
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    
    best_val = -np.inf
    best_result = None
    
    # Try various aspect ratios
    # w + h = 2 (perimeter 4)
    # Try w from 0.5 to 1.5
    w_values = np.linspace(0.5, 1.5, 11)
    
    for w in w_values:
        if _elapsed() > TIME_LIMIT:
            break
        h = 2.0 - w
        
        # Try multiple initializations
        inits = []
        
        # Hexagonal
        for seed in range(3):
            inits.append(_init_hexagonal(w, h, 21, seed=seed))
        
        # Grid configurations
        for rows in [3, 4, 5, 6]:
            for cols in [4, 5, 6, 7]:
                if rows * cols >= 21:
                    inits.append(_init_grid(w, h, 21, rows, cols, seed=0))
        
        # Run optimization from each init
        for idx, init_pts in enumerate(inits):
            if _elapsed() > TIME_LIMIT:
                break
            
            # Force-directed pre-processing
            fd_pts = _force_directed(init_pts, w, h, n_iter=100)
            
            # L-BFGS-B optimization
            pts, val = _optimize_from_init(fd_pts, w, h, seed=idx)
            
            if pts is not None and val > best_val:
                best_val = val
                best_result = (w, h, pts)
            
            # Simulated annealing refinement
            if pts is not None:
                sa_pts, sa_val = _simulated_annealing(pts, w, h, n_iter=300, seed=idx)
                if sa_val > best_val:
                    best_val = sa_val
                    best_result = (w, h, sa_pts)
        
        # Additional L-BFGS-B from best so far for this aspect ratio
        if best_result is not None:
            w_b, h_b, pts_b = best_result
            if w_b == w and h_b == h:
                fd_pts = _force_directed(pts_b, w, h, n_iter=150)
                pts2, val2 = _optimize_from_init(fd_pts, w, h, seed=100)
                if pts2 is not None and val2 > best_val:
                    best_val = val2
                    best_result = (w, h, pts2)
    
    if best_result is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= 21:
                    break
                x = (j + 0.5) / 5
                y = (i + 0.5) / 5
                r = 0.5 / 5 - 0.01
                cells.append([x, y, r])
        return np.array(cells)
    
    w, h, pts = best_result
    
    # Final repair
    pts = _repair(pts, w, h)
    
    return pts
# EVOLVE-BLOCK-END
