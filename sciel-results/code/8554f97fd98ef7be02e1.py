"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _relax(xs, ys, rs, w, h, iters=50):
    """
    Simple force-based relaxation to fix overlaps and expand circles.
    Returns updated xs, ys, rs.
    """
    n = len(rs)
    for _ in range(iters):
        # Check bounds and overlaps
        valid = True
        
        # 1. Push circles away from boundaries
        for i in range(n):
            if xs[i] < rs[i]:
                xs[i] = rs[i]
                valid = False
            if xs[i] > w - rs[i]:
                xs[i] = w - rs[i]
                valid = False
            if ys[i] < rs[i]:
                ys[i] = rs[i]
                valid = False
            if ys[i] > h - rs[i]:
                ys[i] = h - rs[i]
                valid = False
        
        # 2. Push circles away from each other
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                
                if dist < min_dist:
                    valid = False
                    if dist < 1e-8:
                        # If coincident, push randomly
                        angle = np.random.rand() * 2 * np.pi
                        push = (min_dist - dist) / 2
                        xs[i] += np.cos(angle) * push
                        ys[i] += np.sin(angle) * push
                        xs[j] -= np.cos(angle) * push
                        ys[j] -= np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2
                        unit_x = dx / dist
                        unit_y = dy / dist
                        xs[i] += unit_x * push
                        ys[i] += unit_y * push
                        xs[j] -= unit_x * push
                        ys[j] -= unit_y * push
        
        if valid:
            break
            
    # After relaxation, try to expand radii slightly if space permits
    # We can do a simple greedy expansion
    for _ in range(10):
        expanded = False
        for i in range(n):
            # Max possible radius limited by bounds
            max_r_bound = min(xs[i], w - xs[i], ys[i], h - ys[i])
            if rs[i] < max_r_bound:
                # Check against neighbors
                can_expand = True
                new_r = rs[i]
                for j in range(n):
                    if i == j: continue
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Max r such that dist >= r + rs[j]
                    r_limit = dist - rs[j]
                    if r_limit < new_r:
                        new_r = r_limit
                        if r_limit <= 0:
                            can_expand = False
                            break
                if can_expand and new_r > rs[i] * 1.01: # Expand if significant gain
                    # Ensure we don't exceed bounds
                    if new_r <= max_r_bound:
                        rs[i] = new_r
                        expanded = True
        
        if not expanded:
            break
            
    return xs, ys, rs

def _greedy_grid_pack(w, h, n_circles, grid_res=50, rng=None):
    """
    Greedy selection of circles from a grid.
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    # Create grid of potential centers
    xs_grid = np.linspace(0, w, grid_res)
    ys_grid = np.linspace(0, h, grid_res)
    
    # We want to maximize sum of radii.
    # Heuristic: Larger circles are better.
    # Let's try placing circles in order of "potential" radius.
    # Potential radius at a point is limited by distance to boundary and other circles.
    
    # Initial state: empty
    centers = []
    radii = []
    
    # To make it deterministic but varied, we can shuffle the grid points slightly
    # Or just iterate. Let's iterate but prioritize points closer to center?
    # Actually, for packing, corners are often good for large circles, but center is good for density.
    # Let's just try to place circles one by one, picking the position that allows the largest radius.
    
    # Pre-compute all grid points
    points = []
    for x in xs_grid:
        for y in ys_grid:
            points.append([x, y])
    points = np.array(points)
    
    # Shuffle points to avoid bias
    indices = rng.permutation(len(points))
    points = points[indices]
    
    for _ in range(n_circles):
        best_point = None
        best_radius = 0
        
        for p in points:
            x, y = p
            # Max radius from boundaries
            r_bound = min(x, w-x, y, h-y)
            if r_bound <= 0:
                continue
            
            # Check against existing circles
            min_dist_to_existing = np.inf
            for cx, cy, cr in zip(centers, radii, radii):
                d = np.sqrt((x-cx)**2 + (y-cy)**2)
                r_limit = d - cr
                if r_limit < min_dist_to_existing:
                    min_dist_to_existing = r_limit
            
            r_possible = min(r_bound, min_dist_to_existing)
            
            if r_possible > best_radius:
                best_radius = r_possible
                best_point = p
        
        if best_point is not None and best_radius > 0:
            centers.append(best_point[0])
            radii.append(best_radius)
        else:
            break
            
    if len(centers) < n_circles:
        return None
        
    xs = np.array(centers)
    ys = np.array([c[1] for c in centers])
    rs = np.array(radii)
    
    return xs, ys, rs

def _validate(xs, ys, rs, w, h, tol=1e-5):
    """Check if configuration is valid."""
    n = len(rs)
    for i in range(n):
        if xs[i] - rs[i] < -tol or xs[i] + rs[i] > w + tol:
            return False
        if ys[i] - rs[i] < -tol or ys[i] + rs[i] > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Try a range of aspect ratios
    # Perimeter = 2*(w+h) = 4 => w+h = 2
    # Let w vary from 0.5 to 1.5 (since h = 2-w, h must be > 0.5 for symmetry, but let's check wider range)
    # w in [0.5, 1.5] covers square (1,1) and rectangles like (0.5, 1.5) and (1.5, 0.5)
    
    rng = np.random.RandomState(123)
    
    # Define aspect ratios to try
    # w = 2 / (1 + aspect)
    # aspect = h/w
    # If w=1, h=1, aspect=1
    # If w=0.5, h=1.5, aspect=3
    # If w=1.5, h=0.5, aspect=1/3
    
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.333]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 / (1.0 + aspect)
        h = 2.0 - w
        
        # Try multiple grid resolutions and random seeds for the grid shuffle
        for grid_res in [40, 60, 80]:
            if time.time() > deadline:
                break
                
            # We can also try different "seeds" for the random permutation of grid points
            for seed in [42, 123, 999]:
                if time.time() > deadline:
                    break
                    
                rng_sub = np.random.RandomState(seed)
                result = _greedy_grid_pack(w, h, N, grid_res=grid_res, rng=rng_sub)
                
                if result is None:
                    continue
                    
                xs, ys, rs = result
                
                # Relax to fix any small overlaps and expand
                xs_rel, ys_rel, rs_rel = _relax(xs, ys, rs, w, h, iters=100)
                
                if _validate(xs_rel, ys_rel, rs_rel, w, h):
                    value = np.sum(rs_rel)
                    if value > best_value:
                        best_value = value
                        best_result = np.column_stack([xs_rel, ys_rel, rs_rel])
                        
    # If no valid result found via grid, use a simple fallback
    if best_result is None:
        # Simple grid fallback
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
        # Relax this too
        xs = np.array([c[0] for c in cells])
        ys = np.array([c[1] for c in cells])
        rs = np.array([c[2] for c in cells])
        xs, ys, rs = _relax(xs, ys, rs, w, h, iters=50)
        best_result = np.column_stack([xs, ys, rs])
        
    return best_result
# EVOLVE-BLOCK-END
