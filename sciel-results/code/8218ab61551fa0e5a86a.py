import time
import random
import sys
from math import gcd
from itertools import combinations

# Deadline guard: kill after 140s to stay under 150s evaluator limit
_DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    # Map from line (normalized) to list of points on it
    # Or simply count collinear points for each pair
    # For N=20, O(N^3) is 8000 ops, very fast.
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_triple_masks(points):
    """
    Build a list of collinear triples.
    Returns a list of frozensets of 3 indices.
    Also returns a list of masks (integers) for each point representing 
    which other points it is collinear with? No, that's for pairs.
    For the scorer, we need the list of forbidden triples.
    """
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            line_points = [i, j]
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    line_points.append(k)
            m = len(line_points)
            if m >= 3:
                # Any 3 of these form a collinear triple
                for a in range(m):
                    for b in range(a + 1, m):
                        for c in range(b + 1, m):
                            triple = frozenset([line_points[a], line_points[b], line_points[c]])
                            if triple not in seen:
                                seen.add(triple)
                                triples.append(triple)
    return triples

def _max_independent_set_size(triples, n):
    """
    Exact branch and bound for maximum subset with no three collinear.
    Uses bitmask representation for speed.
    """
    if not triples:
        return n

    # Precompute for each index, the set of triples it participates in
    # And for each pair, the set of third points that complete a triple?
    
    # Let's use a recursive B&B with bitmasks.
    # candidates: bitmask of points that can still be added
    # current: bitmask of points already selected
    # We want to maximize popcount(current)
    
    # To prune, we can use a simple upper bound: popcount(candidates) + popcount(current)
    
    # Better pruning: 
    # If we select point p, we must remove from candidates any point q 
    # such that there exists r in current where {p, q, r} is a triple.
    
    # Let's precompute for each p, and for each r, the mask of q's that conflict.
    # conflict[p][r] = mask of q such that {p, q, r} is a triple.
    # Note: p, q, r are distinct.
    
    conflict = [[0] * n for _ in range(n)]
    for t in triples:
        p, q, r = list(t)
        # p conflicts with q given r
        conflict[p][r] |= (1 << q)
        conflict[p][q] |= (1 << r)
        # q conflicts with p given r
        conflict[q][r] |= (1 << p)
        conflict[q][p] |= (1 << r)
        # r conflicts with p given q
        conflict[r][p] |= (1 << q)
        conflict[r][q] |= (1 << p)

    best = [0]
    
    # Order indices by degree (number of triples they are in) to branch on high-degree nodes first?
    # Or just iterate 0..n-1? 
    # Standard MIS B&B: pick a vertex, branch include/exclude.
    # Here, we can just iterate through points in a fixed order, but use the conflict masks to update candidates.
    
    def backtrack(candidates, count, pos):
        if _elapsed() > _DEADLINE - 5:
            return
        
        if candidates == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound
        # Simple bound: count + popcount(candidates)
        # Python 3.10+ has int.bit_count()
        if count + candidates.bit_count() <= best[0]:
            return
        
        # Pick the next point to consider
        # We can pick the lowest set bit for determinism, or highest degree.
        # Let's pick the lowest set bit for simplicity, as the conflict update is the main cost.
        # However, branching on a point that conflicts with many remaining candidates is better.
        
        # Find the point in candidates with the highest "conflict potential"
        # Or just take the first one?
        # For N=20, taking the first one is often fast enough if bounds are tight.
        
        # Let's find the index of the lowest set bit
        p = (candidates & -candidates).bit_length() - 1
        
        # Branch 1: Exclude p
        backtrack(candidates & ~(1 << p), count, pos + 1)
        
        # Branch 2: Include p
        # Update candidates: remove p, and remove any q that conflicts with p given any r in current?
        # Wait, the state needs to know `current` to compute the new candidates.
        # My previous `backtrack` signature didn't have `current`.
        # We need `current` to determine which conflicts are active.
        
        # Let's rewrite backtrack to take `current` mask.
        
        # Actually, we can just pass `current` and `candidates`.
        # When we include p:
        # new_current = current | (1 << p)
        # new_candidates = candidates & ~(1 << p)
        # For each r in current:
        #    new_candidates &= ~conflict[p][r]
        # Note: conflict[p][r] contains q such that {p,q,r} is a triple.
        # If r is in current, and we add p, we cannot add any q that forms a triple with p and r.
        
        # So we need to iterate through bits of `current`.
        
        # Let's modify the function signature.
        pass

    # Re-implementing with `current`
    def backtrack(current, candidates, count):
        if _elapsed() > _DEADLINE - 5:
            return
        
        if candidates == 0:
            if count > best[0]:
                best[0] = count
            return
        
        # Upper bound
        if count + candidates.bit_count() <= best[0]:
            return
        
        # Pick a point p from candidates
        # Heuristic: pick p that maximizes the reduction in candidates?
        # Or just the first one.
        p = (candidates & -candidates).bit_length() - 1
        
        # Exclude p
        backtrack(current, candidates & ~(1 << p), count)
        
        # Include p
        new_candidates = candidates & ~(1 << p)
        # Apply conflicts with all r in current
        curr_mask = current
        while curr_mask:
            r = (curr_mask & -curr_mask).bit_length() - 1
            new_candidates &= ~conflict[p][r]
            curr_mask &= curr_mask - 1
            
        backtrack(current | (1 << p), new_candidates, count + 1)

    backtrack(0, (1 << n) - 1, 0)
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _build_triple_masks(points)
    return _max_independent_set_size(triples, len(points))

def _random_config():
    """Generate a random valid configuration."""
    points = []
    seen = set()
    attempts = 0
    while len(points) < 20 and attempts < 5000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        key = (x, y)
        if key not in seen:
            seen.add(key)
            points.append([x, y])
        attempts += 1
    
    if len(points) != 20:
        return None
    
    if not _check_no_four_collinear(points):
        return None
        
    return points

def _perturb(points):
    """Perturb one point to a random new position."""
    new_points = [list(p) for p in points]
    idx = random.randint(0, 19)
    # Try to find a valid new position
    for _ in range(50):
        nx = random.randint(-100, 100)
        ny = random.randint(-100, 100)
        new_points[idx] = [nx, ny]
        
        # Check distinctness
        valid_distinct = True
        seen = set()
        for p in new_points:
            key = tuple(p)
            if key in seen:
                valid_distinct = False
                break
            seen.add(key)
        
        if not valid_distinct:
            continue
            
        # Check no four collinear
        if _check_no_four_collinear(new_points):
            return new_points
            
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_score = 21
    
    random.seed(42)
    
    # 1. Try to find a good starting configuration via random restarts
    # We want a config with LOW score.
    # Random configs usually have high score (few collinearities).
    # We want MANY collinearities to FORCE triples.
    # So random start is bad. We need a structured start.
    
    # Strategy: Start with a grid-like pattern that has many 3-collinearities.
    # Example: Points on lines y = x, y = -x, y = 2x, etc.
    # But we must avoid 4-collinear.
    
    # Let's try a "star" pattern or multiple lines with 3 points each.
    # If we have 6 lines, each with 3 points, and they intersect nicely, we get many triples.
    # Total points 20.
    
    # Let's try a specific known good heuristic:
    # Place points on a small grid, then perturb to remove 4-collinear.
    
    # Grid 5x4 = 20 points.
    # Points: (i, j) for i in 0..4, j in 0..3.
    # This has many 4-collinear (rows, cols).
    # We need to break them.
    
    # Let's just use random search but with a bias towards collinearity?
    # No, let's use the "two parabola" idea from the current program as a seed, 
    # but improve the search.
    
    # Actually, the cell finding says "Greedy Construction that maximizes collinear triples".
    # Let's implement a simple greedy:
    # 1. Start with 20 random points.
    # 2. For each point, try to move it to a position that creates a new triple 
    #    with two existing points, without creating a 4-collinear.
    #    If such a move reduces the score, take it.
    
    # However, scoring is expensive.
    # Let's just do local search on random starts, but with a better scorer.
    
    # To get a better start, let's try a few structured patterns.
    
    # Pattern 1: 3 points on each of 6 lines, plus 2 extra?
    # Let's try lines:
    # y = 0: (0,0), (1,0), (2,0) -> 3 collinear
    # y = 1: (0,1), (1,1), (2,1) -> 3 collinear
    # ...
    # This creates 4-collinear in columns.
    
    # Let's stick to Random + Local Search, but run it for longer with the fast scorer.
    # The fast scorer (bitmask B&B) should be very fast for N=20.
    
    for restart in range(100):
        if _elapsed() > _DEADLINE - 20:
            break
            
        # Generate a random start
        config = _random_config()
        if config is None:
            continue
            
        s = _score(config)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in config]
            
        # Local search
        current = [list(p) for p in config]
        current_score = s
        stall_count = 0
        
        while _elapsed() < _DEADLINE - 10:
            perturbed = _perturb(current)
            if perturbed is None:
                stall_count += 1
                if stall_count > 50:
                    break
                continue
            
            # Check if the move actually changed anything
            if perturbed == current:
                stall_count += 1
                if stall_count > 50:
                    break
                continue
                
            s_new = _score(perturbed)
            if s_new < current_score:
                current = perturbed
                current_score = s_new
                best_points = [list(p) for p in current]
                best_score = s_new
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 100:
                    break
                    
    if best_points is None:
        # Fallback: return a valid random config
        best_points = _random_config()
        if best_points is None:
             # Absolute fallback: 20 points on a line? No, 4-collinear forbidden.
             # Just return a grid with perturbations
             pts = []
             for i in range(5):
                 for j in range(4):
                     pts.append([i*10, j*10 + i]) # Shear to avoid 4-collinear?
             # Check validity
             if _check_no_four_collinear(pts):
                 best_points = pts
             else:
                 # Random fallback
                 best_points = _random_config()
                 
    return best_points

# Initialize start time
_start_time = time.time()
