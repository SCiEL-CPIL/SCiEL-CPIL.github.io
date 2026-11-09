"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 26

# EVOLVE-BLOCK-START

def _max_radius_at(pos, others):
    """Calculate the maximum radius a circle at 'pos' can have without overlapping others or boundaries."""
    x, y = pos
    # Distance to boundaries
    r = min(x, y, 1.0 - x, 1.0 - y)
    if r <= 0:
        return 0.0
    
    for (ox, oy, orr) in others:
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        # The new circle must not overlap with the existing one.
        # dist >= r + orr  =>  r <= dist - orr
        max_r_constraint = dist - orr
        if max_r_constraint < r:
            r = max_r_constraint
    if r < 0:
        return 0.0
    return r

def _score(circles):
    """Calculate the sum of radii."""
    return np.sum(circles[:, 2])

def _is_valid(circles, tol=1e-8):
    """Check if circles are inside the square and disjoint."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            return False
        if r < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _init_hexagonal(n):
    """Initialize circles in a hexagonal grid pattern."""
    circles = []
    # Approximate number of rows
    rows = int(np.sqrt(n * 2 / np.sqrt(3)))
    cols = int(np.ceil(n / rows))
    
    # Generate hex grid points
    points = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5 * (i % 2)) / (cols + 0.5)
            y = (i + 0.5) / (rows + 0.5)
            # Scale to fit in [0.1, 0.9] roughly to allow growth
            x = 0.1 + 0.8 * x
            y = 0.1 + 0.8 * y
            points.append((x, y))
            
    # Shuffle and take n
    random.shuffle(points)
    points = points[:n]
    
    # Start with small radii
    for x, y in points:
        r = 0.01
        circles.append([x, y, r])
        
    return np.array(circles)

def _greedy_expand(circles, max_iter=1000, stuck_threshold=1e-5):
    """
    Greedy expansion: repeatedly pick the circle with the most clearance and expand it.
    If stuck (no clearance), jiggle a random circle to escape local minimum.
    """
    n = len(circles)
    c = [list(circ) for circ in circles]
    
    for _ in range(max_iter):
        # Find circle with max clearance
        best_idx = -1
        best_gain = -1.0
        
        for i in range(n):
            x, y, r = c[i]
            others = [c[j] for j in range(n) if j != i]
            max_r = _max_radius_at((x, y), others)
            gain = max_r - r
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain <= stuck_threshold:
            # Stuck! Jiggle a random circle to break symmetry/constraint
            # Pick a random circle, shrink it slightly and move it
            idx = random.randint(0, n - 1)
            x, y, r = c[idx]
            
            # Shrink to allow movement
            new_r = max(0.01, r * 0.9)
            
            # Try to find a new position for this circle that allows a larger radius
            # We'll just move it to a random position and let the next iteration's expansion fix it
            # Or better: move it to a position that maximizes radius given OTHERS are fixed
            # But that's expensive. Let's just jitter it.
            
            # Jitter position
            new_x = x + random.uniform(-0.1, 0.1)
            new_y = y + random.uniform(-0.1, 0.1)
            
            # Clamp to boundaries with margin
            new_x = max(new_r, min(1.0 - new_r, new_x))
            new_y = max(new_r, min(1.0 - new_r, new_y))
            
            c[idx] = [new_x, new_y, new_r]
            
            # After jiggle, we might have created overlaps with others.
            # We need to resolve overlaps for this specific circle.
            # Simple resolution: push this circle away from others.
            for _ in range(10):
                moved = False
                for j in range(n):
                    if j == idx: continue
                    xi, yi, ri = c[idx]
                    xj, yj, rj = c[j]
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    min_dist = ri + rj
                    if dist < min_dist - 1e-9:
                        if dist < 1e-9:
                            angle = random.uniform(0, 2 * np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                        else:
                            dx = (xi - xj) / dist
                            dy = (yi - yj) / dist
                        overlap = min_dist - dist
                        c[idx][0] += dx * overlap
                        c[idx][1] += dy * overlap
                        moved = True
                if not moved: break
                
            # Clamp again
            x, y, r = c[idx]
            c[idx][0] = max(r, min(1.0 - r, c[idx][0]))
            c[idx][1] = max(r, min(1.0 - r, c[idx][1]))
            
            # If we still can't expand, try again in next loop
            continue
            
        # Expand the best circle
        x, y, r = c[best_idx]
        # We expand to the max possible radius
        c[best_idx][2] = r + best_gain
        
        # Expanding might cause overlaps? 
        # _max_radius_at ensures no overlap with others AT THE CURRENT POSITION.
        # So expanding to max_radius_at is safe regarding overlaps with others.
        # It is safe regarding boundaries by definition.
        # So no collision resolution needed for the expansion itself!
        # This is much faster than LS.
        
    return np.array(c)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_circles = None
    best_score = -1.0
    
    # Try multiple random seeds for initialization
    for seed in range(50):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize
        circles = _init_hexagonal(N)
        
        # Run greedy expansion
        # We run for a fixed number of steps, or until no improvement
        # The _greedy_expand function handles the "stuck" logic internally
        expanded = _greedy_expand(circles, max_iter=500)
        
        # Validate
        if _is_valid(expanded):
            score = _score(expanded)
            if score > best_score:
                best_score = score
                best_circles = expanded.copy()
                
    # If no valid circles found, fallback to simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        
    return best_circles
# EVOLVE-BLOCK-END
