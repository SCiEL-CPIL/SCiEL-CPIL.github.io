"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _compute_score(circles):
    """Compute sum of radii if valid, else 0.0."""
    if circles.shape[0] != N:
        return 0.0
    if np.any(circles[:, 2] <= 0):
        return 0.0
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Check walls
    if np.any(cx - r < -1e-9) or np.any(cx + r > 1e9):
        return 0.0
    # We need to find the bounding box
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return 0.0
    perim = 2 * (w + h)
    if perim > 4 + 1e-7:
        return 0.0
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return 0.0
    return np.sum(r)

def _normalize_and_score(circles):
    """Normalize circles to fit in perimeter-4 rect, return score."""
    if circles.shape[0] != N:
        return 0.0, None
    cx, cy, r = circles[:, 0].copy(), circles[:, 1].copy(), circles[:, 2].copy()
    if np.any(r <= 0):
        return 0.0, None
    
    # Get bounding box
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    w = max_x - min_x
    h = max_y - min_y
    if w <= 0 or h <= 0:
        return 0.0, None
    
    # Scale to perimeter 4
    scale = 2.0 / (w + h)
    cx = (cx - min_x) * scale
    cy = (cy - min_y) * scale
    r = r * scale
    
    # Now w' = 2, h' = 2 - w' ? No.
    # New w = w * scale, New h = h * scale
    # Perimeter = 2(w*scale + h*scale) = 2*scale*(w+h) = 2*(2) = 4. Correct.
    
    # Check overlaps in scaled space
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return 0.0, None
            
    # Check walls (should be tight)
    if np.any(cx - r < -1e-7) or np.any(cx + r > w * scale + 1e-7):
        return 0.0, None
    if np.any(cy - r < -1e-7) or np.any(cy + r > h * scale + 1e-7):
        return 0.0, None
        
    return np.sum(r), np.column_stack((cx, cy, r))

def _pbd_solve(circles, aspect_ratio, max_iters=1000):
    """
    Position-Based Dynamics solver.
    circles: (N, 3) array [cx, cy, r]
    aspect_ratio: w/h. We fix height H=1, so W=aspect_ratio.
    Actually, let's fix the box to W x H such that 2(W+H)=4 => W+H=2.
    Let H = 1 / (1 + 1/aspect_ratio) ? No.
    Let H = 2 / (1 + aspect_ratio), W = 2 * aspect_ratio / (1 + aspect_ratio).
    """
    W = 2.0 * aspect_ratio / (1.0 + aspect_ratio)
    H = 2.0 / (1.0 + aspect_ratio)
    
    cx = circles[:, 0].copy()
    cy = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Initial scaling to fit in box roughly
    # Find current bbox
    min_x = np.min(cx - r)
    max_x = np.max(cx + r)
    min_y = np.min(cy - r)
    max_y = np.max(cy + r)
    w = max_x - min_x
    h = max_y - min_y
    if w > 0 and h > 0:
        scale = min(W / w, H / h) * 0.9
        cx = (cx - min_x) * scale + (W - w * scale) / 2
        cy = (cy - min_y) * scale + (H - h * scale) / 2
        r = r * scale
        
    # PBD Iterations
    dt = 1.0
    for it in range(max_iters):
        # 1. Predict positions (not strictly needed for static packing, but helps dynamics if we had velocity)
        # For static packing, we just project.
        
        # 2. Resolve Constraints (Projection)
        # We do multiple sub-iterations for convergence
        for sub in range(10):
            # Wall constraints
            for i in range(N):
                # Left
                if cx[i] - r[i] < 0:
                    cx[i] = r[i]
                # Right
                if cx[i] + r[i] > W:
                    cx[i] = W - r[i]
                # Bottom
                if cy[i] - r[i] < 0:
                    cy[i] = r[i]
                # Top
                if cy[i] + r[i] > H:
                    cy[i] = H - r[i]
            
            # Collision constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist < 1e-6:
                            dx = 1.0
                            dy = 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        # Push apart
                        overlap = min_dist - dist
                        # Equal mass assumption
                        cx[i] += dx * overlap * 0.5
                        cy[i] += dy * overlap * 0.5
                        cx[j] -= dx * overlap * 0.5
                        cy[j] -= dy * overlap * 0.5
                        
    return np.column_stack((cx, cy, r))

def _expand_radii(circles, aspect_ratio, max_steps=50):
    """
    Try to increase radii slightly while maintaining feasibility via PBD.
    """
    cx = circles[:, 0].copy()
    cy = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    W = 2.0 * aspect_ratio / (1.0 + aspect_ratio)
    H = 2.0 / (1.0 + aspect_ratio)
    
    for step in range(max_steps):
        # Try to increase all radii by a small factor
        factor = 1.005
        r_new = r * factor
        
        # Check if this is immediately infeasible in terms of box size
        # If any circle is larger than half the box dimension, it's bad.
        if np.any(r_new > W / 2) or np.any(r_new > H / 2):
            break
            
        # Attempt to resolve with PBD using new radii
        # We need a solver that takes radii as fixed and adjusts positions
        # Let's reuse the logic but with fixed r
        cx_curr = cx.copy()
        cy_curr = cy.copy()
        
        feasible = True
        for sub in range(20):
            # Walls
            for i in range(N):
                if cx_curr[i] - r_new[i] < 0: cx_curr[i] = r_new[i]
                if cx_curr[i] + r_new[i] > W: cx_curr[i] = W - r_new[i]
                if cy_curr[i] - r_new[i] < 0: cy_curr[i] = r_new[i]
                if cy_curr[i] + r_new[i] > H: cy_curr[i] = H - r_new[i]
            
            # Collisions
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx_curr[i] - cx_curr[j]
                    dy = cy_curr[i] - cy_curr[j]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    min_dist = r_new[i] + r_new[j]
                    if dist < min_dist:
                        if dist < 1e-6:
                            dx, dy, dist = 1.0, 0.0, 1.0
                        else:
                            dx /= dist; dy /= dist
                        overlap = min_dist - dist
                        cx_curr[i] += dx * overlap * 0.5
                        cy_curr[i] += dy * overlap * 0.5
                        cx_curr[j] -= dx * overlap * 0.5
                        cy_curr[j] -= dy * overlap * 0.5
                        
        # Check final feasibility
        valid = True
        for i in range(N):
            if cx_curr[i] - r_new[i] < -1e-6 or cx_curr[i] + r_new[i] > W + 1e-6:
                valid = False; break
            if cy_curr[i] - r_new[i] < -1e-6 or cy_curr[i] + r_new[i] > H + 1e-6:
                valid = False; break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx_curr[i] - cx_curr[j]
                    dy = cy_curr[i] - cy_curr[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_new[i] + r_new[j] - 1e-6:
                        valid = False; break
                if not valid: break
                
        if valid:
            cx, cy, r = cx_curr, cy_curr, r_new
        else:
            # Reduce step size or break
            break
            
    return np.column_stack((cx, cy, r))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Initial configurations
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    grid_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side) for i in range(side) for j in range(side)]
    init_configs = [np.array(grid_cells[:N])]
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        cx = np.random.rand(N)
        cy = np.random.rand(N)
        r = np.full(N, 0.1)
        init_configs.append(np.column_stack((cx, cy, r)))
        
    # 3. Hexagonal-ish
    hex_configs = []
    for i in range(N):
        row = i // 6
        col = i % 6
        x = col * 0.9 + (0.45 if row % 2 else 0)
        y = row * 0.8
        r = 0.4
        hex_configs.append((x, y, r))
    init_configs.append(np.array(hex_configs))
    
    # Aspect ratios to try
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
        for init in init_configs:
            if time.time() > deadline:
                break
            
            # 1. Solve PBD with initial radii
            solved = _pbd_solve(init, aspect, max_iters=200)
            
            # 2. Expand radii
            expanded = _expand_radii(solved, aspect, max_steps=100)
            
            # 3. Score
            score, norm_circles = _normalize_and_score(expanded)
            
            if score > best_score:
                best_score = score
                best_circles = norm_circles
                
    # If we have time, do a local polish on the best one
    if best_circles is not None and time.time() < deadline - 10:
        # Polish: small random perturbations + PBD + Expand
        cx, cy, r = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        # Infer aspect from bbox
        min_x, max_x = np.min(cx-r), np.max(cx+r)
        min_y, max_y = np.min(cy-r), np.max(cy+r)
        w, h = max_x-min_x, max_y-min_y
        aspect = w/h if h>0 else 1.0
        
        for _ in range(50):
            if time.time() > deadline - 5: break
            # Perturb
            cx_p = cx + np.random.randn(N) * 0.01
            cy_p = cy + np.random.randn(N) * 0.01
            r_p = r.copy()
            
            solved = _pbd_solve(np.column_stack((cx_p, cy_p, r_p)), aspect, max_iters=100)
            expanded = _expand_radii(solved, aspect, max_steps=50)
            
            score, norm_circles = _normalize_and_score(expanded)
            if score > best_score:
                best_score = score
                best_circles = norm_circles
                
    if best_circles is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles

# EVOLVE-BLOCK-END
