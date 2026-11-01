import numpy as np
import time

N = 21
DEADLINE = time.time() + 140.0  # Safety margin before 150s limit

def _force_relax(x, y, r, w, h, max_iter=2000, tol=1e-8):
    """
    Vectorized force-directed relaxation to resolve overlaps and boundary violations.
    Updates x, y in-place.
    """
    n = len(r)
    dt = 0.05
    min_dt = 1e-6
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        # Compute pairwise forces
        # dx[i,j] = x[i] - x[j]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx**2 + dy**2
        # Avoid division by zero
        dist_sq[dist_sq < 1e-12] = 1e-12
        dist = np.sqrt(dist_sq)
        
        r_sum = r[:, None] + r[None, :]
        
        # Overlap amount: positive if overlapping
        overlap = r_sum - dist
        # Only apply force if overlapping
        mask = overlap > 0
        # Force magnitude is proportional to overlap
        # We want to push them apart along the line connecting centers
        # Force on i from j: (overlap / dist) * (dx, dy)
        # But we only want to move if overlapping
        force_x = np.zeros_like(x)
        force_y = np.zeros_like(y)
        
        # Vectorized force accumulation
        # f_ij = overlap_ij * (dx_ij / dist_ij)
        # We need to sum over j for each i
        # Using broadcasting, this creates an (n,n) matrix
        # To save memory/time, we can compute only the upper triangle and add to both
        # But for n=21, full matrix is fine (441 elements)
        
        # Calculate force components
        # Note: dist is 0 on diagonal, but overlap is 0 there too (r_i + r_i - 0 is not 0, wait)
        # Diagonal: dist=0, overlap = 2r_i. This is a problem.
        # We must exclude diagonal.
        np.fill_diagonal(dist_sq, 1.0) # Prevent div by 0
        np.fill_diagonal(dist, 1.0)
        
        # Recalculate overlap with safe dist
        overlap = r_sum - dist
        mask = overlap > 0
        # Set diagonal to 0
        np.fill_diagonal(overlap, 0.0)
        
        # Force vector components
        fx = overlap * (dx / dist)
        fy = overlap * (dy / dist)
        
        # Sum forces on each particle
        # fx[i,j] is force on i due to j? 
        # dx = x[i] - x[j]. If x[i] > x[j], dx > 0. 
        # If overlapping, we want to push i to the right (positive x) and j to the left (negative x).
        # So force on i is +fx[i,j] and force on j is -fx[i,j].
        # Summing fx over j gives total force on i.
        force_x = np.sum(fx, axis=1)
        force_y = np.sum(fy, axis=1)
        
        # Boundary forces
        # If x[i] < r[i], push right. If x[i] > w - r[i], push left.
        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h
        
        force_x -= left_viol
        force_x += right_viol
        force_y -= bottom_viol
        force_y += top_viol
        
        # Update positions
        # Adaptive step size: reduce if moving too much
        max_move = np.max(np.sqrt(force_x**2 + force_y**2))
        if max_move > 1e-6:
            step = min(dt, 0.1 / max_move)
        else:
            step = dt
            
        x += step * force_x
        y += step * force_y
        
        # Convergence check
        if max_move < tol:
            break
            
    # Final clip to ensure strict boundary adherence
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    return x, y

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if configuration is strictly valid."""
    if np.any(r <= 0):
        return False
    if w <= 0 or h <= 0:
        return False
    if 2*(w+h) > 4.0 + tol:
        return False
        
    # Boundary check
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
        
    # Overlap check
    n = len(r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    # Check all pairs i < j
    # dist_sq[i,j] >= r_sum[i,j]^2
    # We can just check if min(dist_sq - r_sum^2) > -tol
    diff = dist_sq - r_sum**2
    # Ignore diagonal
    np.fill_diagonal(diff, 1.0)
    
    if np.min(diff) < -tol:
        return False
        
    return True

def _grow_and_pack(w, h, rng, max_growth_iters=500):
    """
    Greedy growth algorithm.
    1. Initialize small circles in a grid or random positions.
    2. Relax positions.
    3. Grow radii.
    4. Repeat.
    """
    # Initialize positions: Grid based for stability
    # 5x5 grid = 25 spots, take 21
    side = 5
    cols = 5
    rows = 5
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Fill grid
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            x[idx] = (j + 0.5) / cols * w
            y[idx] = (i + 0.5) / rows * h
            r[idx] = min(w/cols, h/rows) * 0.4 # Start small
            idx += 1
            
    # Add some randomness to break symmetry
    x += rng.uniform(-0.05*w, 0.05*w, N)
    y += rng.uniform(-0.05*h, 0.05*h, N)
    r += rng.uniform(0, 0.01, N)
    
    # Ensure initial feasibility
    x = np.clip(x, r, w-r)
    y = np.clip(y, r, h-r)
    
    current_value = np.sum(r)
    
    for it in range(max_growth_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # 1. Relax positions to remove overlaps
        x, y = _force_relax(x, y, r, w, h, max_iter=100)
        
        # 2. Try to grow radii
        # Estimate how much we can grow
        # Find the most constrained pair or boundary
        # Simple heuristic: try to grow all by a small factor
        # Or calculate max possible growth based on current overlaps if any remain (should be 0 after relax)
        
        # Calculate min distance to any other circle or boundary
        # This is expensive to do exactly, but we can approximate or use a binary search on growth factor
        
        # Let's try a fixed small growth step and check validity
        growth_factor = 1.005
        new_r = r * growth_factor
        
        # Check if new_r is feasible with current x,y
        # If yes, accept. If no, try smaller growth.
        # For speed, we just check validity.
        if _is_valid(x, y, new_r, w, h, tol=1e-9):
            r = new_r
            current_value = np.sum(r)
        else:
            # Binary search for max growth factor?
            # Or just stop growing if we can't grow at all
            # Let's try a smaller step
            if growth_factor <= 1.001:
                break
            growth_factor = 1.002
            new_r = r * growth_factor
            if _is_valid(x, y, new_r, w, h, tol=1e-9):
                r = new_r
                current_value = np.sum(r)
            else:
                # Can't grow anymore
                break
                
    # Final polish: Relax positions again to ensure max value is maintained
    x, y = _force_relax(x, y, r, w, h, max_iter=500)
    
    # Final validity check
    if _is_valid(x, y, r, w, h, tol=1e-9):
        return x, y, r, current_value
    else:
        # Shrink radii slightly to ensure validity
        for scale in [0.99, 0.98, 0.95, 0.9]:
            r_trial = r * scale
            if _is_valid(x, y, r_trial, w, h, tol=1e-9):
                return x, y, r_trial, np.sum(r_trial)
        return None, None, None, 0.0

def construct():
    start_time = time.time()
    best_value = 0.0
    best_circles = None
    
    # Try different aspect ratios
    # Perimeter 4 => w + h = 2
    # w in [0.5, 1.5] roughly
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1, 0.6, 1.4, 0.85, 1.15, 0.75, 1.25]
    
    # Shuffle to vary order
    rng = np.random.default_rng(42)
    rng.shuffle(aspect_ratios)
    
    # Also try some random aspect ratios
    for _ in range(10):
        aspect_ratios.append(rng.uniform(0.5, 1.5))
        
    for w in aspect_ratios:
        if time.time() > DEADLINE - 20:
            break
            
        h = 2.0 - w
        if h < 0.3 or w < 0.3:
            continue
            
        # Multiple random restarts for this aspect ratio
        for restart in range(3):
            if time.time() > DEADLINE - 15:
                break
                
            x, y, r, val = _grow_and_pack(w, h, rng)
            
            if x is not None and val > best_value:
                best_value = val
                best_circles = np.column_stack((x, y, r))
                
    if best_circles is None:
        # Fallback: Simple grid in square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (j + 0.5) / side
                y = (i + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
        best_circles = np.array(cells[:N])
        
    return best_circles
