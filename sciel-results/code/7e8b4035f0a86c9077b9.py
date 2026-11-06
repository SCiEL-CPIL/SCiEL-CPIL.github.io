"""Improved program: search for 20 integer points with small no-3-collinear subset."""
import time
import random
from itertools import combinations
from collections import defaultdict

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 115  # safety margin before 120s limit


def _elapsed():
    return time.time() - _START_TIME


def _timeout():
    return _elapsed() > _TIME_LIMIT


def _count_collinear_triples(points):
    """Count number of collinear triples in a set of points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Check other points on this line
            count = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Cross product check: (x3-x1)*dy - (y3-y1)*dx == 0
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    count += 1
            if count >= 1:
                triples += count
    return triples // 2  # Each triple counted once per pair... actually let me recount


def _count_3collinear(points):
    """Count distinct collinear triples."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    triples += 1
    return triples


def _has_4_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                count = 3
                for l in range(k + 1, n):
                    x4, y4 = points[l]
                    if (x4 - x1) * dy - (y4 - y1) * dx == 0:
                        count += 1
                if count >= 4:
                    return True
    return False


def _largest_no_3_collinear(points):
    """
    Find the largest subset of points with no three collinear.
    Uses branch and bound with bitmasks for efficiency.
    """
    n = len(points)
    
    # Precompute collinearity: collinearity[i] is a bitmask of points collinear with i
    collinearity = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                if (x3 - x1) * dy - (y3 - y1) * dx == 0:
                    collinearity[i] |= (1 << j) | (1 << k)
                    collinearity[j] |= (1 << i) | (1 << k)
                    collinearity[k] |= (1 << i) | (1 << j)
    
    best = [0]
    
    def search(remaining_mask, current_size, last_added):
        """
        remaining_mask: bitmask of points that can still be added
        current_size: size of current valid subset
        last_added: index of last point added (-1 if none)
        """
        if _timeout():
            return
        
        # Update best
        if current_size > best[0]:
            best[0] = current_size
        
        # If no remaining points, stop
        if remaining_mask == 0:
            return
        
        # Pruning: even if we add all remaining, can't beat best
        max_possible = current_size + bin(remaining_mask).count('1')
        if max_possible <= best[0]:
            return
        
        # Greedy upper bound: try to see how many more we can add
        # Simple bound: at most remaining count
        # Better bound: each new point conflicts with some remaining
        
        # Try adding each candidate
        candidates = []
        mask = remaining_mask
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            candidates.append(idx)
            mask ^= bit
        
        # Sort candidates by how few conflicts they have (heuristic)
        def conflict_count(idx):
            # How many points in remaining_mask are collinear with idx
            conflicts = 0
            for c in candidates:
                if c == idx:
                    continue
                if c < n and (collinearity[idx] >> c) & 1:
                    conflicts += 1
            return conflicts
        
        # For efficiency, just iterate
        for idx in candidates:
            if _timeout():
                return
            
            # Check if idx conflicts with current subset
            # A point conflicts if it's collinear with any two points already in subset
            # Actually: adding idx is valid if no two points in current subset are collinear with idx
            # i.e., for all pairs (a,b) in current subset, not (a,b,idx collinear)
            # Equivalently: for all a in current subset, (collinearity[idx] & (1<<a)) should have at most... 
            # Wait: idx creates a triple if there exist two other points a,b in subset such that a,b,idx are collinear.
            # That means: the number of points in current subset that are collinear with idx must be <= 1.
            
            # Count how many points in current subset are collinear with idx
            # We track current subset implicitly... let me restructure
            
            # Actually, let's track the current subset as a set/list
            pass
    
    # Let me restructure with explicit subset tracking
    best[0] = 0
    
    def search2(subset, remaining_mask, depth):
        if _timeout():
            return
        
        if len(subset) > best[0]:
            best[0] = len(subset)
        
        if remaining_mask == 0:
            return
        
        # Upper bound
        if len(subset) + bin(remaining_mask).count('1') <= best[0]:
            return
        
        # Try adding each point from remaining
        candidates = []
        mask = remaining_mask
        while mask:
            bit = mask & (-mask)
            idx = bit.bit_length() - 1
            candidates.append(idx)
            mask ^= bit
        
        for idx in candidates:
            if _timeout():
                return
            
            # Check validity: no two points in subset are collinear with idx
            # i.e., for any pair (a,b) in subset, idx is not on line ab
            # Equivalently: at most 1 point in subset is collinear with idx
            collinear_with_subset = 0
            for a in subset:
                if (collinearity[idx] >> a) & 1:
                    collinear_with_subset += 1
                    if collinear_with_subset > 1:
                        break
            
            if collinear_with_subset > 1:
                continue
            
            # Also check: adding idx shouldn't create a triple among subset+{idx}
            # This is the same condition: no two in subset collinear with idx
            
            new_remaining = remaining_mask & ~(1 << idx)
            # Remove points that would conflict with future additions? No, just remove idx
            search2(subset + [idx], new_remaining, depth + 1)
    
    search2([], (1 << n) - 1, 0)
    return best[0]


def _score(points):
    """Compute the score: largest subset with no three collinear."""
    return _largest_no_3_collinear(points)


def _valid(points):
    """Check validity: 20 distinct integer points, |x|,|y| <= 100, no 4 collinear."""
    if len(points) != 20:
        return False
    if len(set(tuple(p) for p in points)) != 20:
        return False
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 100 or abs(y) > 100:
            return False
    if _has_4_collinear(points):
        return False
    return True


def _generate_initial():
    """Generate a valid initial configuration."""
    # Start with parabola points and modify to break up large no-3-collinear subsets
    # Strategy: place points on a few lines so that many triples are collinear
    # This makes it harder to find large no-3-collinear subsets
    
    # Place points on a grid-like structure with deliberate collinearities
    # Use points on lines: several lines with 2-3 points each
    
    # Try: place points in clusters on different lines
    # Line 1: y=0, points at x=-8,-6
    # Line 2: y=1, points at x=-7,-5
    # etc. But need to ensure no 4 collinear
    
    # Better: use a structured approach
    # Place 10 points on y=0 (but that's 10 collinear -> violates no-4)
    # So max 3 per line.
    
    # Use 7 lines with 3 points each (21 points) minus 1, or mix
    # Actually with 20 points and max 3 per line, we need at least ceil(20/3)=7 lines
    # But lines can share points...
    
    # Let's use a different approach: start from parabola and swap some points
    # to introduce collinear triples
    
    base = [[x, x * x] for x in range(-10, 10)]
    
    # Swap some points to create collinear triples
    # For example, replace point (0,0) with a point that's collinear with two others
    # (1,1) and (-1,1) are on y=x^2... no.
    # (2,4) and (-2,4): line through them is y=4. Add (0,4)? 
    # But (0,0) is in base. Replace (0,0) with (0,4).
    # Now (-2,4), (0,4), (2,4) are collinear.
    
    # Let's do several such swaps
    swaps = {
        0: [0, 4],    # replaces (-10, 100)? No, index 0 is x=-10, y=100
    }
    
    # Let me index properly: base[i] = [x, x^2] for x in range(-10,10)
    # i=0: x=-10, y=100
    # i=10: x=0, y=0
    # i=12: x=2, y=4
    # i=8: x=-2, y=4
    
    # Replace point at index 10 (0,0) with (0,4) to make (-2,4),(0,4),(2,4) collinear
    points = list(base)
    points[10] = [0, 4]
    
    # Check validity
    if _valid(points):
        return points
    
    # If not valid, try random search
    for _ in range(10000):
        pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
        if _valid(pts):
            return pts
    
    return base  # fallback


def _local_search(points, time_budget):
    """
    Local search: try to improve the configuration by swapping points.
    Objective: minimize the largest no-3-collinear subset size.
    """
    current_points = [list(p) for p in points]
    current_score = _score(current_points)
    best_points = [list(p) for p in points]
    best_score = current_score
    
    all_x = list(range(-100, 101))
    all_y = list(range(-100, 101))
    
    iteration = 0
    while not _timeout():
        iteration += 1
        improved = False
        
        # Try swapping one point with a random alternative
        for _ in range(500):  # trials per outer iteration
            if _timeout():
                break
            
            idx = random.randint(0, 19)
            old_point = current_points[idx]
            
            # Try several random replacements
            for _ in range(20):
                if _timeout():
                    break
                
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_point = [new_x, new_y]
                
                # Check distinctness
                if any(p == new_point for i, p in enumerate(current_points) if i != idx):
                    continue
                
                current_points[idx] = new_point
                
                if not _valid(current_points):
                    current_points[idx] = old_point
                    continue
                
                new_score = _score(current_points)
                
                if new_score < best_score:
                    best_score = new_score
                    best_points = [list(p) for p in current_points]
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_points[idx] = old_point
        
        if not improved and iteration > 50:
            # Stagnation: try a bigger perturbation
            for _ in range(50):
                if _timeout():
                    break
                idx1 = random.randint(0, 19)
                idx2 = random.randint(0, 19)
                if idx1 == idx2:
                    continue
                
                old1 = current_points[idx1][:]
                old2 = current_points[idx2][:]
                
                current_points[idx1] = [random.randint(-100, 100), random.randint(-100, 100)]
                current_points[idx2] = [random.randint(-100, 100), random.randint(-100, 100)]
                
                if not _valid(current_points):
                    current_points[idx1] = old1
                    current_points[idx2] = old2
                    continue
                
                new_score = _score(current_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [list(p) for p in current_points]
                    current_score = new_score
                    break
                else:
                    current_points[idx1] = old1
                    current_points[idx2] = old2
    
    return best_points, best_score


def construct():
    """
    Return exactly 20 distinct integer points [x, y] with |x|, |y| <= 100 and no four collinear.
    Minimize the largest subset with no three collinear.
    """
    random.seed(42)
    
    # Generate initial valid configuration
    points = _generate_initial()
    
    if not _valid(points):
        # Fallback: try harder
        for _ in range(50000):
            if _timeout():
                break
            pts = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
            if _valid(pts):
                points = pts
                break
    
    if not _valid(points):
        # Last resort: use parabola (guaranteed valid)
        points = [[x, x * x] for x in range(-10, 10)]
    
    # Local search to improve
    best_points, best_score = _local_search(points, _TIME_LIMIT - _elapsed())
    
    # Final validation
    if not _valid(best_points):
        best_points = points
    
    return best_points
