"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_ordinary_lines_fast(points):
    """
    Count lines containing exactly 2 points in O(N^2) time.
    For each point, we count how many other points lie on each line passing through it.
    If a line through point i has k other points, it has k+1 total points.
    If k+1 == 2 (i.e., k == 1), the line is ordinary.
    Each ordinary line is counted twice (once for each endpoint), so we divide by 2 at the end.
    """
    n = len(points)
    if n < 2:
        return 0
    
    ordinary_count = 0
    
    for i in range(n):
        x1, y1 = points[i]
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Canonicalize direction
            if dx == 0 and dy == 0:
                continue # Should not happen for distinct points
            
            g = _gcd(abs(dx), abs(dy))
            dx_n = dx // g
            dy_n = dy // g
            
            # Normalize sign: ensure first non-zero is positive
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            
            key = (dx_n, dy_n)
            slopes[key] = slopes.get(key, 0) + 1
        
        # If a slope has count 1, it means there is exactly 1 other point on that line.
        # So the line has 1 (current) + 1 (other) = 2 points.
        for count in slopes.values():
            if count == 1:
                ordinary_count += 1
                
    return ordinary_count // 2

def _is_valid(points):
    """Check if points are valid: 12 distinct integer points, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) >= 3:
        x0, y0 = points[0]
        x1, y1 = points[1]
        dx = x1 - x0
        dy = y1 - y0
        all_collinear = True
        for x, y in points[2:]:
            if (x - x0) * dy - (y - y0) * dx != 0:
                all_collinear = False
                break
        if all_collinear:
            return False
    return True

def _generate_candidates():
    """Generate a list of candidate point sets with high symmetry."""
    candidates = []
    
    # 1. 3x4 Grid
    pts = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts)
    
    # 2. 4x3 Grid
    pts = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts)
    
    # 3. 2x6 Grid
    pts = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(pts)
    
    # 4. 6x2 Grid
    pts = [[x, y] for x in range(2) for y in range(6)]
    candidates.append(pts)
    
    # 5. Hexagonal Lattice Subset (Center + Ring 1 + Ring 2 partial)
    # Let's try a specific hexagonal arrangement
    # (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1) ... 
    # Standard hex lattice coords: (i, j) -> (i + j/2, j*sqrt(3)/2). 
    # Integer approximations for hex lattice:
    # Let's use a known good configuration for N=12 if possible.
    # The 3x4 grid is often optimal or near-optimal for small N.
    
    # 6. Affine Plane AG(2,3) has 9 points. Let's add 3 more.
    # AG(2,3) points: {(x,y) | x,y in {0,1,2}}
    # Add (3,0), (0,3), (3,3)?
    base_ag23 = [[x, y] for x in range(3) for y in range(3)]
    extra1 = [[3, 0], [0, 3], [3, 3]]
    pts = base_ag23 + extra1
    candidates.append(pts)
    
    # 7. Another variation: 3x3 grid + (3,1), (1,3), (2,2) is already in grid.
    # Let's try a "plus" shape extended?
    
    # 8. Random high-symmetry: 12 points on a circle? No, integer coords.
    # 9. 12 points from a 4x4 grid, removing 4 corners?
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    # Remove (0,0), (0,3), (3,0), (3,3)
    pts = [p for p in grid_4x4 if p not in [[0,0], [0,3], [3,0], [3,3]]]
    candidates.append(pts)
    
    # 10. 12 points from a 5x5 grid?
    # Let's stick to the most promising ones.
    
    return candidates

def _local_search(points, max_iters=1000):
    """
    Simple local search using the fast counter.
    Tries to swap a point with a nearby integer point to reduce ordinary lines.
    """
    current = [list(p) for p in points]
    current_cost = _count_ordinary_lines_fast(current)
    best = [list(p) for p in current]
    best_cost = current_cost
    start_time = time.time()
    deadline = start_time + 10 # 10 seconds max for local search
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Try moving one point
        k = random.randint(0, 11)
        old_x, old_y = current[k]
        
        # Try a few random nearby points
        for _ in range(20):
            nx = random.randint(old_x - 5, old_x + 5)
            ny = random.randint(old_y - 5, old_y + 5)
            if (nx, ny) == (old_x, old_y):
                continue
            
            # Check if point already exists
            exists = False
            for i in range(12):
                if current[i] == [nx, ny]:
                    exists = True
                    break
            if exists:
                continue
            
            current[k] = [nx, ny]
            if _is_valid(current):
                new_cost = _count_ordinary_lines_fast(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = [list(p) for p in current]
                    improved = True
                    break
            current[k] = [old_x, old_y]
            
        if not improved:
            # Reset if no improvement in this iteration? 
            # Or just continue. 
            pass
            
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    random.seed(42)
    
    best_points = None
    best_cost = float('inf')
    
    # Generate and evaluate candidates
    candidates = _generate_candidates()
    
    for cand in candidates:
        if not _is_valid(cand):
            continue
        cost = _count_ordinary_lines_fast(cand)
        if cost < best_cost:
            best_cost = cost
            best_points = [list(p) for p in cand]
            
    # If we have a good candidate, try to improve it with local search
    if best_points is not None:
        # Run local search on the best candidate
        improved_points, improved_cost = _local_search(best_points, max_iters=2000)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_points = improved_points
            
    # If time remains, try a few random restarts?
    # The fast counter is O(N^2) = 144 ops. Very fast.
    # We can afford more local search steps.
    while time.time() < deadline - 5:
        # Perturb best solution
        perturbed = [list(p) for p in best_points]
        for _ in range(random.randint(1, 3)):
            k = random.randint(0, 11)
            # Move to a random integer coordinate nearby
            ox, oy = perturbed[k]
            perturbed[k] = [ox + random.randint(-10, 10), oy + random.randint(-10, 10)]
        
        if not _is_valid(perturbed):
            continue
            
        # Run short local search
        new_points, new_cost = _local_search(perturbed, max_iters=500)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points
            
    if best_points is None:
        # Fallback: 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points

# EVOLVE-BLOCK-END

