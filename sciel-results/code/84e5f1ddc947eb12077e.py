"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
        
    # Check non-overlapping
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _init_grid(n):
    """
    Initialize circles in a grid pattern.
    This provides a good, valid starting point that is not too dense.
    """
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                # Start with small radii to ensure validity
                r = 0.5 / side * 0.8 
                cells.append([x, y, r])
    return np.array(cells[:n])

def _init_random_valid(n, seed=None):
    """
    Initialize with random non-overlapping small circles.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    circles = []
    attempts = 0
    while len(circles) < n and attempts < 1000:
        attempts += 1
        # Sample position
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        
        # Determine max possible radius at this location given existing circles
        r_max = min(x, 1-x, y, 1-y)
        for cx, cy, cr in circles:
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            r_limit = dist - cr - 0.01 # small buffer
            if r_limit < r_max:
                r_max = r_limit
        
        if r_max > 0.01:
            # Pick a radius slightly less than max to ensure safety, or random small
            r = random.uniform(0.02, max(0.03, r_max * 0.8))
            circles.append([x, y, r])
            
    # If we didn't place all, pad with very small circles at random valid spots
    while len(circles) < n:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        circles.append([x, y, 0.01])
        
    return np.array(circles)

def _ls_expand(circles, max_iters=5000, time_limit=None):
    """
    Lubachevsky-Steampole style iterative expansion.
    Repeatedly pick the circle with the smallest "clearance" (max possible radius increase
    limited by neighbors and walls) and expand it to that limit.
    This drives the system towards a "jammed" state where many circles are touching.
    """
    n = len(circles)
    if time_limit is None:
        time_limit = time.time() + 10 # Default 10s per run
    
    # Copy to avoid modifying input
    c = circles.copy()
    
    for _ in range(max_iters):
        if time.time() > time_limit:
            break
            
        # Find the circle with the smallest clearance
        min_clearance = np.inf
        min_idx = -1
        
        for i in range(n):
            x, y, r = c[i]
            
            # Boundary constraints
            r_max = min(x, 1-x, y, 1-y)
            
            # Neighbor constraints
            for j in range(n):
                if i == j:
                    continue
                dx = x - c[j, 0]
                dy = y - c[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_limit = dist - c[j, 2]
                if r_limit < r_max:
                    r_max = r_limit
            
            clearance = r_max - r
            if clearance < min_clearance:
                min_clearance = clearance
                min_idx = i
        
        if min_clearance < 1e-12:
            break # Fully jammed
            
        # Expand the chosen circle
        x, y, r = c[min_idx]
        r_max = min(x, 1-x, y, 1-y)
        for j in range(n):
            if min_idx == j:
                continue
            dx = x - c[j, 0]
            dy = y - c[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_limit = dist - c[j, 2]
            if r_limit < r_max:
                r_max = r_limit
        
        c[min_idx, 2] = r_max
        
    return c

def _gradient_polish(circles, max_iters=2000, time_limit=None):
    """
    Gradient-based local search to maximize sum of radii.
    Uses a penalty method for constraints.
    """
    if time_limit is None:
        time_limit = time.time() + 10
        
    n = len(circles)
    c = circles.copy()
    
    # We optimize positions (x, y) and radii (r)
    # Variables: [x0, y0, r0, x1, y1, r1, ...]
    # But it's easier to keep structure as (n, 3)
    
    lr_pos = 0.005
    lr_rad = 0.005
    penalty_weight = 10.0
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
            
        grad = np.zeros_like(c)
        
        # Compute objective gradient (sum of radii)
        grad[:, 2] = 1.0
        
        # Compute penalty gradients for overlaps and boundaries
        for i in range(n):
            xi, yi, ri = c[i]
            
            # Boundary penalties
            # If xi < ri, push right. If xi > 1-ri, push left.
            # Soft boundary: penalty = max(0, ri - xi)^2 + max(0, ri - (1-xi))^2 ...
            # Gradient w.r.t xi: -2*max(0, ri-xi) + 2*max(0, ri-(1-xi))
            # Gradient w.r.t ri: 2*max(0, ri-xi) - 2*max(0, ri-(1-xi)) - 2*max(0, ri-yi) + 2*max(0, ri-(1-yi))
            
            b_x_left = max(0, ri - xi)
            b_x_right = max(0, ri - (1 - xi))
            b_y_bottom = max(0, ri - yi)
            b_y_top = max(0, ri - (1 - yi))
            
            grad[i, 0] += -2 * b_x_left + 2 * b_x_right
            grad[i, 1] += -2 * b_y_bottom + 2 * b_y_top
            grad[i, 2] += 2 * b_x_left - 2 * b_x_right - 2 * b_y_bottom + 2 * b_y_top
            
            # Overlap penalties
            for j in range(i + 1, n):
                xj, yj, rj = c[j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy) + 1e-12
                overlap = ri + rj - dist
                if overlap > 0:
                    # Penalty = overlap^2
                    # Grad w.r.t xi: 2*overlap * (dx/dist)
                    # Grad w.r.t xj: 2*overlap * (-dx/dist)
                    # Grad w.r.t ri: 2*overlap
                    # Grad w.r.t rj: 2*overlap
                    
                    factor = 2 * overlap / dist
                    grad[i, 0] += factor * dx
                    grad[i, 1] += factor * dy
                    grad[i, 2] += 2 * overlap
                    
                    grad[j, 0] -= factor * dx
                    grad[j, 1] -= factor * dy
                    grad[j, 2] += 2 * overlap
        
        # Apply gradient ascent
        # Separate position and radius updates to control scales
        c[:, 0] += lr_pos * penalty_weight * grad[:, 0]
        c[:, 1] += lr_pos * penalty_weight * grad[:, 1]
        c[:, 2] += lr_rad * grad[:, 2]
        
        # Ensure radii are positive
        c[:, 2] = np.maximum(c[:, 2], 1e-6)
        
        # Project back to valid space if significantly invalid (hard constraint repair)
        # This is a simple repair: shrink any circle that overlaps or goes out of bounds
        valid = True
        for i in range(n):
            if c[i, 0] < c[i, 2] or c[i, 0] > 1 - c[i, 2] or c[i, 1] < c[i, 2] or c[i, 1] > 1 - c[i, 2]:
                valid = False
                break
            for j in range(i + 1, n):
                dist = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                if dist < c[i, 2] + c[j, 2]:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
             # If invalid, we need to repair. 
             # Simple repair: shrink all radii by a small factor until valid
             shrink_factor = 0.95
             while not _is_valid(c) and shrink_factor > 0.1:
                 c[:, 2] *= shrink_factor
                 shrink_factor *= 0.9
             
             # If still invalid, revert to previous valid state? 
             # For simplicity, we just accept the shrunken state and continue.
             # A more robust approach would store last_valid and restore if stuck.

    return c

def construct():
    deadline = time.time() + 140 # Safety margin for 150s limit
    
    best_circles = None
    best_score = -1
    
    # Generate diverse initializations
    # 1. Grid
    # 2. Random
    # 3. Random with different seed
    # 4. "Corner" biased random (circles tend to pack in corners)
    
    def _init_corner_biased(n, seed):
        random.seed(seed)
        np.random.seed(seed)
        circles = []
        # Place a few large circles in corners first
        corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
        random.shuffle(corners)
        for cx, cy in corners[:4]:
            circles.append([cx, cy, 0.15])
            
        # Fill rest randomly
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                r_limit = dist - cr - 0.02
                if r_limit < r_max:
                    r_max = r_limit
            if r_max > 0.02:
                r = random.uniform(0.03, max(0.04, r_max * 0.9))
                circles.append([x, y, r])
            else:
                # Force place small circle
                circles.append([x, y, 0.02])
        
        return np.array(circles[:n])

    starts = [
        _init_grid(N),
        _init_random_valid(N, 42),
        _init_random_valid(N, 123),
        _init_corner_biased(N, 777)
    ]
    
    for start_circles in starts:
        if time.time() > deadline - 30:
            break
            
        # Ensure start is valid
        if not _is_valid(start_circles):
             # Repair
             for _ in range(10):
                 start_circles[:, 2] *= 0.9
                 if _is_valid(start_circles):
                     break
             if not _is_valid(start_circles):
                 continue

        current_circles = start_circles.copy()
        
        # Phase 1: LS Expansion
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        t_ls = min(10, remaining * 0.3)
        t_grad = min(20, remaining * 0.6)
        
        if t_ls > 1:
            c_ls = _ls_expand(current_circles, time_limit=time.time() + t_ls)
            if _is_valid(c_ls) and _sum_radii(c_ls) > _sum_radii(current_circles):
                current_circles = c_ls
        
        # Phase 2: Gradient Polish
        if time.time() < deadline - 5:
            c_grad = _gradient_polish(current_circles, time_limit=time.time() + t_grad)
            if _is_valid(c_grad) and _sum_radii(c_grad) > _sum_radii(current_circles):
                current_circles = c_grad
                
        # Phase 3: Final LS Expansion to squeeze out any last gains
        if time.time() < deadline - 5:
             c_final = _ls_expand(current_circles, time_limit=time.time() + 5)
             if _is_valid(c_final) and _sum_radii(c_final) > _sum_radii(current_circles):
                 current_circles = c_final

        score = _sum_radii(current_circles)
        if score > best_score:
            best_score = score
            best_circles = current_circles.copy()

    if best_circles is None:
        best_circles = _init_grid(N)
        
    return best_circles
# EVOLVE-BLOCK-END
