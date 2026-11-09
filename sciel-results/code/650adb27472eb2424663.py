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
    n = len(circles)
    
    # Check bounds
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
    
    # Check non-overlapping
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


def _max_radius_at(x, y, circles, exclude_idx=None):
    """Calculate the maximum radius a circle at (x,y) can have given other circles."""
    max_r = min(x, 1 - x, y, 1 - y)
    if max_r < 0:
        return 0.0
        
    n = len(circles)
    for j in range(n):
        if exclude_idx is not None and j == exclude_idx:
            continue
        dx = x - circles[j, 0]
        dy = y - circles[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # The new circle must not overlap with circle j
        # dist >= r_new + r_j  =>  r_new <= dist - r_j
        max_r = min(max_r, dist - circles[j, 2])
        
    return max(0.0, max_r)


def _greedy_expand(circles, max_iter=2000, deadline=None):
    """
    Lubachevsky-Steampole style expansion.
    Repeatedly pick a circle, expand it to its maximum possible radius,
    and push neighbors away if they overlap.
    """
    circles = circles.copy()
    n = len(circles)
    
    # Ensure initial validity by shrinking if necessary
    # This is a simple repair: if invalid, shrink all slightly and move to center
    if not _is_valid(circles):
         # Crude repair: move to random valid positions with tiny radius
         x = np.random.uniform(0.1, 0.9, n)
         y = np.random.uniform(0.1, 0.9, n)
         r = np.full(n, 0.01)
         circles = np.column_stack([x, y, r])
         
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Pick a random circle to expand
        i = np.random.randint(n)
        
        # Try to expand circle i
        current_r = circles[i, 2]
        new_r = _max_radius_at(circles[i, 0], circles[i, 1], circles, exclude_idx=i)
        
        if new_r > current_r + 1e-12:
            circles[i, 2] = new_r
            
            # Check for overlaps caused by this expansion
            # If overlap, push the other circle away (or both?)
            # Lubachevsky-Steampole usually pushes the *other* circle away if possible,
            # or moves the expanded circle back. 
            # A simpler variant: if expansion causes overlap, we revert or push neighbor.
            # Let's try: if expansion causes overlap with j, move j away.
            
            # Check validity
            if not _is_valid(circles):
                # Revert radius
                circles[i, 2] = current_r
                
                # Try to push a neighbor away to make room?
                # Or just jiggle?
                # For now, let's implement a simple push-back for the expanded circle
                # or a push-away for neighbors.
                
                # Identify the most conflicting neighbor
                x_i, y_i = circles[i, 0], circles[i, 1]
                max_conflict = -1
                conflict_j = -1
                for j in range(n):
                    if j == i: continue
                    dx = x_i - circles[j, 0]
                    dy = y_i - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    overlap = circles[i, 2] + circles[j, 2] - dist
                    if overlap > max_conflict:
                        max_conflict = overlap
                        conflict_j = j
                        
                if conflict_j != -1 and max_conflict > 1e-9:
                    # Push conflict_j away from i
                    dx = circles[conflict_j, 0] - circles[i, 0]
                    dy = circles[conflict_j, 1] - circles[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-10:
                        ux = dx / dist
                        uy = dy / dist
                        # Move j by overlap amount
                        circles[conflict_j, 0] += ux * max_conflict
                        circles[conflict_j, 1] += uy * max_conflict
                        
                        # Clip to bounds
                        circles[conflict_j, 0] = np.clip(circles[conflict_j, 0], 0, 1)
                        circles[conflict_j, 1] = np.clip(circles[conflict_j, 1], 0, 1)
                        
                        # If moving j caused it to go out of bounds or overlap others, 
                        # we might need to shrink j or revert. 
                        # To keep it simple and robust, if the push makes j invalid, 
                        # we shrink j slightly.
                        if not _is_valid(circles):
                             # Shrink j
                             circles[conflict_j, 2] *= 0.9
                             # If still invalid, revert the push
                             if not _is_valid(circles):
                                  circles[conflict_j, 0] -= ux * max_conflict
                                  circles[conflict_j, 1] -= uy * max_conflict
                                  circles[conflict_j, 2] /= 0.9
                                  # Revert i expansion
                                  # (Already reverted above)
                                  pass
                                  
    return circles


def _local_jiggle(circles, strength=0.01, deadline=None):
    """Apply random perturbations to escape local minima."""
    circles = circles.copy()
    n = len(circles)
    for _ in range(50): # Fixed small number of jiggles
        if deadline is not None and time.time() > deadline:
            break
        i = np.random.randint(n)
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        circles[i, 0] += strength * np.cos(angle)
        circles[i, 1] += strength * np.sin(angle)
        # Clip to bounds
        circles[i, 0] = np.clip(circles[i, 0], 0, 1)
        circles[i, 1] = np.clip(circles[i, 1], 0, 1)
        # If this caused overlap, shrink the circle
        if not _is_valid(circles):
            circles[i, 2] *= 0.95
            if not _is_valid(circles):
                # Revert
                circles[i, 0] -= strength * np.cos(angle)
                circles[i, 1] -= strength * np.sin(angle)
                circles[i, 2] /= 0.95
    return circles


def _init_grid(n):
    """Initialize with a grid."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n: break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells[:n])


def _init_random(n, seed=None):
    """Initialize with random valid circles."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    circles = np.zeros((n, 3))
    # Place first circle
    circles[0] = [0.5, 0.5, 0.1]
    
    for i in range(1, n):
        placed = False
        for attempt in range(100):
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            r = _max_radius_at(x, y, circles[:i])
            if r > 0.01: # Minimum size threshold
                circles[i] = [x, y, r]
                placed = True
                break
        if not placed:
            # Fallback: tiny circle
            circles[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.01]
            
    return circles


def construct():
    """Construct optimal circle packing in unit square."""
    deadline = time.time() + 110  # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Strategy 1: Grid Init
    init = _init_grid(N)
    if _is_valid(init):
        optimized = _greedy_expand(init, max_iter=1000, deadline=deadline - 20)
        if _is_valid(optimized):
            score = _sum_radii(optimized)
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
        
        # Jiggle and expand again
        if time.time() < deadline - 20:
            jiggled = _local_jiggle(optimized, strength=0.05, deadline=deadline - 10)
            if _is_valid(jiggled):
                optimized2 = _greedy_expand(jiggled, max_iter=1000, deadline=deadline - 10)
                if _is_valid(optimized2):
                    score = _sum_radii(optimized2)
                    if score > best_score:
                        best_score = score
                        best_circles = optimized2.copy()
                        
    # Strategy 2: Random Inits
    np.random.seed(42)
    random.seed(42)
    
    n_restarts = 5
    for k in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        init = _init_random(N, seed=42 + k)
        if not _is_valid(init):
            continue
            
        # Expand
        optimized = _greedy_expand(init, max_iter=1000, deadline=deadline - 10)
        if _is_valid(optimized):
            score = _sum_radii(optimized)
            if score > best_score:
                best_score = score
                best_circles = optimized.copy()
                
        # Jiggle
        if time.time() < deadline - 10:
            jiggled = _local_jiggle(optimized, strength=0.05, deadline=deadline - 5)
            if _is_valid(jiggled):
                optimized2 = _greedy_expand(jiggled, max_iter=500, deadline=deadline - 5)
                if _is_valid(optimized2):
                    score = _sum_radii(optimized2)
                    if score > best_score:
                        best_score = score
                        best_circles = optimized2.copy()

    if best_circles is None:
        # Fallback: Grid
        best_circles = _init_grid(N)
        
    return best_circles
# EVOLVE-BLOCK-END
