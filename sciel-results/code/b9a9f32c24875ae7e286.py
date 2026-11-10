import time
import random
import sys
from itertools import combinations

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    # Map lines to counts
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize line equation to canonical form
            # Ax + By + C = 0
            # A = dy, B = -dx, C = dx*y1 - dy*x1
            # To make it canonical, divide by gcd and fix sign
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            
            if A == 0 and B == 0:
                continue # Duplicate points, shouldn't happen in valid input
            
            # Find GCD of A, B, C
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Fix sign: first non-zero of A, B should be positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
                
            if lines[key] >= 4:
                return False
    return True

def _max_independent_set(points):
    """
    Exact branch and bound for the largest subset with no three collinear.
    Uses bitmasks for speed.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute conflict masks
    # conflict[i] is a bitmask of points j such that there exists k 
    # such that i, j, k are collinear? 
    # No, the constraint is "no three collinear".
    # This is a 3-uniform hypergraph independent set problem.
    # A set S is valid if it contains no hyperedge (triple).
    
    # Let's precompute for each pair (i, j) the set of k collinear with them.
    # Actually, a simpler representation for backtracking:
    # We build the set incrementally.
    # State: current_set (bitmask), candidates (bitmask)
    # If we add point p to current_set, we must remove from candidates
    # any point q such that there exists r in current_set where p, q, r are collinear.
    # Wait, if p, q, r are collinear, then {p, q, r} is a forbidden triple.
    # So if we have p and r in current_set, we cannot add q.
    
    # Let's precompute `collinear_with[i][j]` as a bitmask of k such that i, j, k are collinear.
    collinear_with = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    mask |= (1 << k)
            collinear_with[i][j] = mask
            collinear_with[j][i] = mask

    best = 0
    
    def backtrack(current_mask, candidates_mask):
        nonlocal best
        
        # Count bits in candidates
        # Python int bit_count is fast
        cand_count = candidates_mask.bit_count()
        cur_count = current_mask.bit_count()
        
        if cur_count + cand_count <= best:
            return
            
        if cand_count == 0:
            if cur_count > best:
                best = cur_count
            return

        # Heuristic: Pick a candidate point
        # To improve pruning, we should pick a point that conflicts with many others?
        # Or just pick the first one.
        # Let's pick the lowest set bit in candidates_mask.
        # This is O(1) if we use bit operations.
        p_idx = (candidates_mask & -candidates_mask).bit_length() - 1
        rest_mask = candidates_mask & ~(1 << p_idx)
        
        # Branch 1: Include p_idx
        # We need to find which points in rest_mask are "blocked" by adding p_idx.
        # A point q is blocked if there exists r in current_mask such that p_idx, q, r are collinear.
        # i.e., q is in collinear_with[p_idx][r] for some r in current_mask.
        
        blocked_mask = 0
        # Iterate through bits in current_mask
        cm = current_mask
        while cm:
            r_idx = (cm & -cm).bit_length() - 1
            blocked_mask |= collinear_with[p_idx][r_idx]
            cm &= cm - 1
            
        new_candidates = rest_mask & ~blocked_mask
        backtrack(current_mask | (1 << p_idx), new_candidates)
        
        # Branch 2: Exclude p_idx
        backtrack(current_mask, rest_mask)

    backtrack(0, (1 << n) - 1)
    return best

def _generate_parabola_config(offset=0, scale=1):
    """Generate points on y = x^2 + offset, scaled."""
    pts = []
    for i in range(20):
        x = i * scale + offset
        y = x * x
        # Keep within bounds [-100, 100]
        if -100 <= x <= 100 and -100 <= y <= 100:
            pts.append([x, y])
    # If we ran out of space, we might need to adjust or use a different curve
    # For n=20, x=0..19 -> y=0..361. y=361 is > 100.
    # So simple parabola y=x^2 won't fit 20 points in [-100,100].
    # We need a smaller curve or different mapping.
    # Let's use a grid-based approach or a smaller parabola.
    # Actually, the task allows any integer points.
    # Let's try to fit 20 points on a parabola within bounds.
    # Max y is 100. x^2 <= 100 -> x in [-10, 10].
    # That's only 21 distinct x values.
    # If we take x = -10, -9, ..., 9, 10, we have 21 points.
    # y values: 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100.
    # Are there 4 collinear points on y=x^2?
    # No, a line intersects a parabola in at most 2 points.
    # So any subset of points on y=x^2 has no 3 collinear points?
    # Wait. If no 3 are collinear, then the largest subset with no 3 collinear is the whole set (20).
    # Score = 20. We want to MINIMIZE the score.
    # So parabola is a BAD strategy for minimizing the score.
    # We want MANY collinear triples.
    # So we want points arranged in lines of 3.
    
    return None

def _generate_grid_config():
    """
    Generate a grid-like configuration with many collinear triples.
    A 4x5 grid has many lines.
    But we need exactly 20 points.
    Let's try a 4x5 grid.
    x in [0, 3], y in [0, 4].
    Points: (x, y) for x in 0..3, y in 0..4.
    Check 4-collinear.
    Horizontal lines: 4 points. Invalid.
    Vertical lines: 5 points. Invalid.
    Diagonals: 4 points (0,0)-(1,1)-(2,2)-(3,3). Invalid.
    So a full grid is invalid.
    
    We need to perturb the grid to break 4-collinear lines while keeping 3-collinear lines.
    Or use a different structure.
    
    Known construction for low independence number:
    Points on a cubic curve? No, cubic can have more intersections.
    
    Let's use a "perturbed grid" or "random grid".
    Start with a 4x5 grid.
    Randomly swap or move points slightly?
    Or just generate random points and filter?
    
    Better strategy:
    Use a set of lines.
    Create a pool of 20 points by selecting intersections of many lines?
    But we need integer points.
    
    Let's try a specific known configuration for small n.
    For n=20, a good configuration might be derived from a finite geometry or specific lattice.
    
    Alternative:
    Use a "random" search but with a strong heuristic:
    1. Start with a random valid config.
    2. Use local search to maximize the number of collinear triples.
    3. This tends to minimize the independence number.
    
    The current program does this but slowly.
    My program will do a more aggressive local search with a faster evaluator.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Strategy:
    # 1. Generate several diverse initial configurations.
    # 2. For each, run a fast local search to maximize collinear triples.
    # 3. Keep the best one found.
    
    best_points = None
    best_score = float('inf')
    
    # Helper to check validity fast
    def is_valid(pts):
        return _check_no_four_collinear(pts)
    
    # Helper to count triples (proxy for score)
    def count_triples(pts):
        n = len(pts)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
        return count
    
    # Local Search
    def local_search(pts, time_limit=10.0):
        current_pts = [p[:] for p in pts]
        current_triples = count_triples(current_pts)
        
        # If invalid, try to fix? Or just reject.
        if not is_valid(current_pts):
            return None, 0
            
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try moving each point to a nearby location
            for i in range(20):
                if time.time() - start > time_limit:
                    break
                ox, oy = current_pts[i]
                # Try a few random moves
                for _ in range(5):
                    nx = ox + random.randint(-5, 5)
                    ny = oy + random.randint(-5, 5)
                    if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                        continue
                    new_p = [nx, ny]
                    if new_p in current_pts:
                        continue
                    current_pts[i] = new_p
                    if is_valid(current_pts):
                        new_triples = count_triples(current_pts)
                        if new_triples > current_triples:
                            current_triples = new_triples
                            improved = True
                            break # Accept first improvement
                        else:
                            current_pts[i] = [ox, oy]
                    else:
                        current_pts[i] = [ox, oy]
                            
        score = _max_independent_set(current_pts)
        return current_pts, score

    # Generate initial configurations
    configs = []
    
    # Config 1: Random
    random.seed(42)
    for _ in range(5):
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        configs.append(pts)
        
    # Config 2: Grid-like
    # 4x5 grid perturbed
    base_grid = [[x, y] for x in range(4) for y in range(5)]
    # Scale and center
    for i in range(20):
        base_grid[i][0] *= 20
        base_grid[i][1] *= 20
    # Add small random noise
    for i in range(20):
        base_grid[i][0] += random.randint(-2, 2)
        base_grid[i][1] += random.randint(-2, 2)
    # Ensure distinct and valid
    valid_grid = []
    for p in base_grid:
        if p not in valid_grid and is_valid(valid_grid + [p]):
            valid_grid.append(p)
    if len(valid_grid) == 20:
        configs.append(valid_grid)
        
    # Config 3: Parabola-like (but we know this has high independence number)
    # Let's try a "line cluster"
    # Group points into lines of 3.
    # 20 points / 3 = 6 lines with 2 leftovers?
    # Or 5 lines of 4? No, 4 is invalid.
    # So we want many lines of 3.
    # Let's generate lines and pick points from them.
    
    # Run local search on each config
    for idx, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        time_limit = min(remaining / (len(configs) - idx), 15.0)
        
        res_pts, score = local_search(cfg, time_limit)
        
        if res_pts is not None:
            if score < best_score:
                best_score = score
                best_points = res_pts
                print(f"Config {idx}: Score {score}", file=sys.stderr)

    # If no valid config found, return a simple valid one
    if best_points is None:
        # Fallback: Random valid
        pts = []
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p not in pts:
                # Check validity
                temp_pts = pts + [p]
                if is_valid(temp_pts):
                    pts.append(p)
        best_points = pts

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_max_independent_set(pts)}")
    print(f"Valid: {_check_no_four_collinear(pts)}")
