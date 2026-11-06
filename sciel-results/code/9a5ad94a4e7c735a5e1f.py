"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 145
    best_pts = None
    best_score = float('inf')
    
    # Try deterministic grid searches
    # A 6x6 grid has 36 points. We need to choose 12.
    # We can prune using symmetry or just iterate if fast enough.
    # 36 choose 12 is ~1.2e9, which is too slow in Python.
    # We need a smarter search or a smaller grid.
    
    # Let's try a 5x5 grid (25 points). 25 choose 12 is ~5.2e6. This is feasible.
    # If we don't find 6, we try 6x6 with a heuristic or smaller subset search.
    
    # Strategy:
    # 1. Try all subsets of size 12 from a 5x5 grid.
    # 2. If best found is 6, return it.
    # 3. If not, try a structured search on 6x6 or use local search from best 5x5.
    
    grid_size = 5
    grid_pts = [[x, y] for x in range(grid_size) for y in range(grid_size)]
    
    # Precompute line keys for all pairs in the grid to speed up counting?
    # No, counting 12 points is O(66) operations. 5e6 * 66 is ~3e8 ops. Might be tight in 150s.
    # Let's optimize the counting.
    
    # Alternative: Use the property that we want few ordinary lines.
    # Most lines should have >= 3 points.
    # In a 5x5 grid, max collinear is 5.
    
    # Let's just run the brute force on 5x5 but with an early termination if we find 6.
    # Also, we can skip subsets that are likely bad?
    
    # Actually, let's try a specific known configuration first if we can derive it.
    # The "Hesse" configuration is for 9 points.
    # For 12 points, the minimum is 6.
    
    # Let's do a randomized search combined with local search, but starting from
    # the best grid configuration found.
    
    # First, quick scan of 5x5 grid subsets.
    # To speed up, we can sample if the full search is too slow.
    # But 5.2e6 is small. Let's try full search with a time check.
    
    start_time = time.time()
    
    # Optimization: Represent points as indices.
    # Precompute lines for the grid?
    # For a subset, we need to count lines with exactly 2 points.
    
    # Let's use a different approach:
    # Iterate through all 12-subsets of the 25 grid points.
    
    # To make it faster, we can use a bitmask or just combinations.
    # combinations(range(25), 12)
    
    # Let's limit the search time for this phase to 30s.
    phase_deadline = time.time() + 30
    
    found_best = None
    found_best_score = float('inf')
    
    # We can prune: if we have a point set, and we add a point, 
    # we can update the count incrementally? 
    # No, removing a point changes many lines.
    
    # Let's just brute force with a fast counter.
    
    def fast_count(pts):
        # pts is list of (x,y)
        n = len(pts)
        line_map = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2
                g = gcd(gcd(abs(a), abs(b)), abs(c))
                if g > 0:
                    a //= g
                    b //= g
                    c //= g
                if a < 0 or (a == 0 and b < 0):
                    a, b, c = -a, -b, -c
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = 0
                line_map[key] += 1
        # Count lines with count == 1 (since each pair adds 1, a line with k points has k(k-1)/2 pairs? No.)
        # My previous logic: line_map[key] was a set of points.
        # Here I am counting pairs? No, I am adding 1 for each pair.
        # If a line has k points, it has C(k,2) pairs.
        # So line_map[key] will be C(k,2).
        # We want lines where the number of points is 2.
        # C(2,2) = 1.
        # So we count keys where value == 1.
        return sum(1 for v in line_map.values() if v == 1)
        
    # Wait, if a line has 3 points, it has 3 pairs. Value = 3.
    # If a line has 4 points, it has 6 pairs. Value = 6.
    # So yes, value == 1 implies exactly 2 points.
    
    # This is much faster than sets.
    
    for combo in combinations(range(25), 12):
        if time.time() > phase_deadline:
            break
        pts = [grid_pts[i] for i in combo]
        # Check collinearity? If all collinear, score is 0? 
        # "Not all collinear" is a constraint.
        # If all collinear, it's invalid.
        # In a 5x5 grid, max collinear is 5. So 12 points cannot be all collinear.
        score = fast_count(pts)
        if score < found_best_score:
            found_best_score = score
            found_best = pts
            if score == 6:
                # Found theoretical minimum
                return [list(p) for p in found_best]
                
    if found_best:
        best_pts = found_best
        best_score = found_best_score
    else:
        # Fallback to 4x3 grid if search failed or didn't run
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
        
    # If we didn't find 6 in 5x5, try local search from best_pts
    # Or try 6x6 grid with random sampling?
    # Let's try a few random samples from 6x6 grid if time permits.
    
    if best_score > 6 and time.time() < deadline - 10:
        grid_size_2 = 6
        grid_pts_2 = [[x, y] for x in range(grid_size_2) for y in range(grid_size_2)]
        
        # Random sampling from 6x6
        import random
        random.seed(42)
        
        samples = 5000
        for _ in range(samples):
            if time.time() > deadline - 5:
                break
            combo = random.sample(range(36), 12)
            pts = [grid_pts_2[i] for i in combo]
            score = fast_count(pts)
            if score < best_score:
                best_score = score
                best_pts = pts
                if score == 6:
                    return [list(p) for p in best_pts]
                    
    # Final polish: Local search from best_pts
    # Use the fast_count for evaluation in local search?
    # Local search moves one point.
    # We can reuse the logic from the seed but with fast_count.
    
    if time.time() < deadline - 5:
        # Simple local search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                orig_x, orig_y = best_pts[i]
                # Try moving to a nearby point in a 6x6 grid?
                # Or just random small moves?
                # Let's try to snap to grid points or nearby integers.
                # Since we are in a grid context, try moving to adjacent grid points
                # or points that create new collinearities.
                
                # For simplicity, try all points in a 10x10 grid around the current point?
                # No, that's 100 moves * 12 points * iterations.
                # Let's just try random moves in a small box.
                
                for _ in range(5):
                    if time.time() > deadline - 2:
                        break
                    dx = random.randint(-5, 5)
                    dy = random.randint(-5, 5)
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = orig_x + dx, orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    
                    # Check distinctness
                    valid = True
                    for j in range(12):
                        if j == i:
                            continue
                        if best_pts[j][0] == new_x and best_pts[j][1] == new_y:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    # Temporarily move
                    best_pts[i] = [new_x, new_y]
                    new_score = fast_count(best_pts)
                    
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        if best_score == 6:
                            return [list(p) for p in best_pts]
                    else:
                        # Revert
                        best_pts[i] = [orig_x, orig_y]
                        
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
