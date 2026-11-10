import random
import time
import sys
from itertools import combinations

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def count_collinear_triples(points):
    """Count the number of collinear triples in the set of points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if invalid."""
    n = len(points)
    # For each pair, count points on the line
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def max_independent_set_3_uniform(points):
    """
    Compute the size of the largest subset of points with no three collinear.
    Uses Branch and Bound with bitmasks.
    """
    n = len(points)
    if n <= 2:
        return n

    # Precompute collinear triples as a list of (i, j, k)
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    if not collinear_triples:
        return n

    # pair_to_third[i][j] = set of k such that {i, j, k} is collinear
    # Using a dictionary of dictionaries for speed
    pair_to_third = {}
    for i in range(n):
        pair_to_third[i] = {}
        for j in range(n):
            if i == j: continue
            pair_to_third[i][j] = set()
            
    for i, j, k in collinear_triples:
        pair_to_third[i][j].add(k)
        pair_to_third[i][k].add(j)
        pair_to_third[j][i].add(k)
        pair_to_third[j][k].add(i)
        pair_to_third[k][i].add(j)
        pair_to_third[k][j].add(i)

    best_size = 0
    full_mask = (1 << n) - 1

    def solve_recur(current_size, current_mask, remaining_mask):
        nonlocal best_size
        
        if not remaining_mask:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Pruning: if current_size + popcount(remaining) <= best_size, prune
        if current_size + bin(remaining_mask).count('1') <= best_size:
            return

        # Pick a point from remaining
        # Heuristic: Pick the point with the most conflicts? 
        # Or just the lowest index.
        p = -1
        for i in range(n):
            if remaining_mask & (1 << i):
                p = i
                break
                
        if p == -1:
            return

        # Option 1: Exclude p
        solve_recur(current_size, current_mask, remaining_mask & ~(1 << p))

        # Option 2: Include p
        # Check if p is compatible with current_mask
        # p is incompatible if there exist r, s in current_mask such that {p, r, s} is collinear.
        # This is equivalent to: for any r in current_mask, if there exists s in current_mask 
        # such that s in pair_to_third[p][r], then incompatible.
        # Actually, if we include p, we must ensure no two points in current_mask form a triple with p.
        # And no point in current_mask forms a triple with p and another point in current_mask.
        
        # Check compatibility with current_mask:
        compatible = True
        # Iterate through points in current_mask
        cm = current_mask
        while cm:
            lsb = cm & -cm
            r = (lsb.bit_length() - 1)
            # If r and p are in a triple with some s in current_mask, it's invalid.
            # We need to check if any s in current_mask is in pair_to_third[p][r]
            # pair_to_third[p][r] is a set of indices.
            # We need to check intersection of pair_to_third[p][r] and current_mask.
            if pair_to_third[p][r] & current_mask:
                compatible = False
                break
            cm ^= lsb
            
        if compatible:
            new_current_mask = current_mask | (1 << p)
            new_remaining = remaining_mask & ~(1 << p)
            
            # Remove forbidden points from new_remaining
            # A point q in new_remaining is forbidden if it forms a triple with p and any r in new_current_mask.
            # Actually, if q is in new_remaining, and we include p, q is forbidden if 
            # there exists r in new_current_mask such that {p, r, q} is collinear.
            # Note: r can be p? No. r is in new_current_mask.
            # So for each r in new_current_mask, q is forbidden if q in pair_to_third[p][r].
            
            forbidden_mask = 0
            # Iterate through points in new_current_mask
            ncm = new_current_mask
            while ncm:
                lsb = ncm & -ncm
                r = (lsb.bit_length() - 1)
                # Add all q in pair_to_third[p][r] to forbidden_mask
                # But only those in new_remaining
                for q in pair_to_third[p][r]:
                    if new_remaining & (1 << q):
                        forbidden_mask |= (1 << q)
                ncm ^= lsb
                
            new_remaining &= ~forbidden_mask
            solve_recur(current_size + 1, new_current_mask, new_remaining)

    solve_recur(0, 0, full_mask)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_triple_count = -1
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    # Pre-generate a pool of candidate points to speed up local search
    # We want points that are likely to form collinear triples
    # but not 4-collinear.
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Strategy: Construct points on a "thick" line or grid to force collinearity
        # but perturb to avoid 4-collinearity.
        
        # 1. Generate base points on a grid
        ox = random.randint(-50, 50)
        oy = random.randint(-50, 50)
        sx = random.randint(5, 20)
        sy = random.randint(5, 20)
        
        candidates = []
        for i in range(5):
            for j in range(5):
                x = ox + i * sx
                y = oy + j * sy
                if -100 <= x <= 100 and -100 <= y <= 100:
                    candidates.append((x, y))
        
        # Add some random points to ensure variety
        for _ in range(10):
            candidates.append(random_point())
        
        random.shuffle(candidates)
        
        # Select 20 distinct points
        points = []
        seen = set()
        for p in candidates:
            if p not in seen:
                seen.add(p)
                points.append(p)
                if len(points) == 20:
                    break
        
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        
        # Repair: Ensure no 4 collinear
        valid = False
        for _ in range(50):
            if not has_four_collinear(points):
                valid = True
                break
            # Find a violating line and move one point
            # Simple repair: move a random point
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        
        if not valid:
            continue
        
        current_triple_count = count_collinear_triples(points)
        
        if current_triple_count > best_triple_count:
            best_triple_count = current_triple_count
            best_points = points.copy()
            
        # Local Search: Maximize collinear triples
        # Use simulated annealing or hill climbing
        current_points = points.copy()
        current_tc = current_triple_count
        
        # Hill climbing with restarts
        for step in range(1000):
            if time.time() > deadline:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            # Try moving the point to a nearby location or random
            if random.random() < 0.5:
                # Move nearby
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_p = (old_p[0] + dx, old_p[1] + dy)
                if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                    continue
            else:
                new_p = random_point()
            
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            if has_four_collinear(candidate_points):
                continue
                
            new_tc = count_collinear_triples(candidate_points)
            
            # Accept if better, or with some probability if worse (annealing-like)
            if new_tc >= current_tc or (new_tc < current_tc and random.random() < 0.1):
                current_points = candidate_points
                current_tc = new_tc
                if current_tc > best_triple_count:
                    best_triple_count = current_tc
                    best_points = current_points.copy()

    if best_points is None:
        # Fallback
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(100):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points

    return [list(p) for p in best_points]
