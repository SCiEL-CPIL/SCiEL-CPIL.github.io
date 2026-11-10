"""Rewritten program: search for 20-point config with no 4 collinear, minimizing max no-3-collinear subset."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
from collections import defaultdict

_DEADLINE = 135.0  # safety margin before 150s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _cross(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def _collinear3(p1, p2, p3):
    return _cross(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]) == 0

def _build_line_map(points):
    """Map each line (as frozenset of point indices) to count of points on it."""
    n = len(points)
    line_to_points = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            # Compute line through i, j in canonical form: ax + by + c = 0
            ax = points[j][1] - points[i][1]
            by = points[i][0] - points[j][0]
            c = -(ax * points[i][0] + by * points[i][1])
            # Normalize
            g = _gcd(abs(ax), abs(by))
            if g > 0:
                ax //= g
                by //= g
                c //= g
            # Ensure canonical sign
            if ax < 0 or (ax == 0 and by < 0) or (ax == 0 and by == 0 and c < 0):
                ax, by, c = -ax, -by, -c
            key = (ax, by, c)
            line_to_points[key].append(i)
            line_to_points[key].append(j)
    # Build unique lines with their point sets
    lines = []
    seen = set()
    for key, idxs in line_to_points.items():
        pt_set = frozenset(idxs)
        if len(pt_set) >= 3:
            lines.append(pt_set)
    return lines

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_collinear_triples(points):
    """Count number of collinear triples in the point set."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    count += 1
    return count

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if (_collinear3(points[i], points[j], points[k]) and
                        _collinear3(points[i], points[j], points[l])):
                        return True
    return False

def _has_four_collinear_fast(points):
    """Faster check using line grouping."""
    n = len(points)
    if n < 4:
        return False
    line_pts = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            ax = points[j][1] - points[i][1]
            by = points[i][0] - points[j][0]
            c = -(ax * points[i][0] + by * points[i][1])
            g = _gcd(abs(ax), abs(by))
            if g > 0:
                ax //= g
                by //= g
                c //= g
            if ax < 0 or (ax == 0 and by < 0) or (ax == 0 and by == 0 and c < 0):
                ax, by, c = -ax, -by, -c
            key = (ax, by, c)
            line_pts[key].append(i)
            line_pts[key].append(j)
    for key, idxs in line_pts.items():
        if len(set(idxs)) >= 4:
            return True
    return False

def _max_independent_set_size(points):
    """
    Exact branch-and-bound for maximum subset with no three collinear.
    Returns the size of the largest such subset.
    """
    n = len(points)
    # Precompute collinear triples
    collinear_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear3(points[i], points[j], points[k]):
                    collinear_triples.append((i, j, k))
    
    if not collinear_triples:
        return n
    
    # For each point, store which triples it's in
    point_to_triples = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(collinear_triples):
        point_to_triples[i].append(t_idx)
        point_to_triples[j].append(t_idx)
        point_to_triples[k].append(t_idx)
    
    # State: which points are included, which triples are violated
    # Use a different approach: branch on points, track which triples are "active"
    # A triple is active if all its points are included
    
    # Greedy upper bound: for remaining points, max we can add is bounded by
    # the number of points not in any active triple... this is complex.
    # Use a simpler bound: total remaining + current size
    
    best = [0]
    
    # Order points by degree (number of collinear triples they participate in)
    order = sorted(range(n), key=lambda i: -len(point_to_triples[i]))
    pos_of = [0] * n
    for idx, p in enumerate(order):
        pos_of[p] = idx
    
    # We'll use a recursive search with memoization on (remaining_points_mask, violated_triples_mask)
    # But that's too large. Instead, use a simpler branch and bound.
    
    # Alternative: since n=20, we can use a different exact method.
    # Let's use the fact that we want max subset with no 3 collinear.
    # This is equivalent to: find largest S such that no triple from collinear_triples is fully in S.
    
    # Use a simple but effective branch and bound:
    # At each step, either include or exclude the next point.
    # Upper bound: current_size + number of remaining points that don't form a violated triple
    # with already-included points.
    
    included = [False] * n
    violated = [False] * len(collinear_triples)
    
    def _upper_bound(start_idx):
        """Upper bound on additional points we can include from start_idx onward."""
        # Simple bound: all remaining points
        # Better: for each remaining point, check if including it would violate any triple
        # with already-included points. If so, it might still be includable if the other
        # two aren't included. This is complex, so use a loose bound.
        count = 0
        for idx in range(start_idx, n):
            p = order[idx]
            if not included[p]:
                # Check if p can be added without violating any triple with included points
                can_add = True
                for t_idx in point_to_triples[p]:
                    i, j, k = collinear_triples[t_idx]
                    # Count how many of i,j,k are already included
                    inc_count = (1 if included[i] else 0) + (1 if included[j] else 0) + (1 if included[k] else 0)
                    if inc_count == 2:
                        # Adding p would make this triple fully included
                        can_add = False
                        break
                if can_add:
                    count += 1
        return count
    
    def search(start_idx, current_size):
        if _time_left() < 2:
            return
        if current_size + _upper_bound(start_idx) <= best[0]:
            return
        if start_idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        p = order[start_idx]
        
        # Option 1: exclude p
        search(start_idx + 1, current_size)
        
        # Option 2: include p (if it doesn't violate any triple)
        can_include = True
        for t_idx in point_to_triples[p]:
            i, j, k = collinear_triples[t_idx]
            if included[i] and included[j]:
                can_include = False
                break
            if included[i] and included[k]:
                can_include = False
                break
            if included[j] and included[k]:
                can_include = False
                break
        
        if can_include:
            included[p] = True
            # Mark triples that become violated (all three included)
            newly_violated = []
            for t_idx in point_to_triples[p]:
                i, j, k = collinear_triples[t_idx]
                if included[i] and included[j] and included[k]:
                    if not violated[t_idx]:
                        violated[t_idx] = True
                        newly_violated.append(t_idx)
            
            # Wait, if we include p and it completes a triple, that's a violation.
            # But we already checked can_include which checks if any triple would be completed.
            # So if can_include is True, no triple is completed. So newly_violated should be empty.
            # Let me re-examine.
            
            # Actually, can_include checks if including p would make any triple fully included.
            # If can_include is True, then no such triple exists, so we're safe.
            
            search(start_idx + 1, current_size + 1)
            
            included[p] = False
    
    search(0, 0)
    return best[0]

def _max_independent_set_size_fast(points):
    """
    Faster exact computation using a different strategy.
    For n=20, we can enumerate all subsets of size k (from 20 down) and check.
    But that's 2^20 = 1M, which is feasible.
    """
    n = len(points)
    
    # Precompute collinearity for all triples
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    collinear[i][j][k] = True
                    collinear[i][k][j] = True
                    collinear[j][i][k] = True
                    collinear[j][k][i] = True
                    collinear[k][i][j] = True
                    collinear[k][j][i] = True
    
    # Check if a subset (bitmask) has no three collinear
    def has_no_three_collinear(mask):
        # Get list of points in mask
        pts = [i for i in range(n) if (mask >> i) & 1]
        for idx in range(len(pts)):
            for jdx in range(idx+1, len(pts)):
                for kdx in range(jdx+1, len(pts)):
                    if collinear[pts[idx]][pts[jdx]][pts[kdx]]:
                        return False
        return True
    
    # Binary search on the answer
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        # Check if there exists a subset of size mid with no three collinear
        found = False
        # Use itertools.combinations for small mid
        if mid <= 15:
            for combo in itertools.combinations(range(n), mid):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                if has_no_three_collinear(mask):
                    found = True
                    break
                if _time_left() < 5:
                    return mid  # Return current best estimate
        else:
            # For large mid, it's harder. Use a heuristic.
            # If mid is large, likely no such subset exists unless very few collinear triples.
            # Try a few random subsets
            for _ in range(100):
                subset = random.sample(range(n), mid)
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                if has_no_three_collinear(mask):
                    found = True
                    break
                if _time_left() < 5:
                    return mid
        if found:
            lo = mid
        else:
            hi = mid - 1
    return lo

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    return _max_independent_set_size_fast(points)

def _validate(points):
    """Check that points are valid: 20 distinct, |x|,|y|<=100, no four collinear."""
    if len(points) != 20:
        return False
    if len(set(tuple(p) for p in points)) != 20:
        return False
    for x, y in points:
        if abs(x) > 100 or abs(y) > 100:
            return False
    if _has_four_collinear_fast(points):
        return False
    return True

def _initial_parabola():
    """Generate 20 points on a parabola."""
    return [[x, x * x] for x in range(-10, 10)]

def _initial_dense():
    """Generate a dense initial configuration with many collinearities."""
    # Use a grid-like pattern with some perturbation
    points = []
    # Try to create many 3-point lines
    # Use points on several lines
    lines = [
        # Horizontal lines
        [(-10, 0), (-5, 0), (0, 0), (5, 0), (10, 0)],
        # Vertical lines  
        [(0, -10), (0, -5), (0, 5), (0, 10)],
        # Diagonal
        [(-10, -10), (-5, -5), (0, 0), (5, 5), (10, 10)],
        # Anti-diagonal
        [(-10, 10), (-5, 5), (5, -5), (10, -10)],
        # Other lines
        [(-8, -4), (-4, 0), (0, 4), (4, 8)],
        [(8, -4), (4, 0), (0, 4), (-4, 8)],
    ]
    for line in lines:
        for p in line:
            if p not in points:
                points.append(list(p))
    # Add more points to reach 20
    if len(points) < 20:
        # Add random points
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in points:
                points.append([x, y])
    return points[:20]

def _local_search(points, max_iters=10000):
    """Local search: swap points to reduce the score."""
    current = [list(p) for p in points]
    if not _validate(current):
        return None, None
    
    current_score = _score(current)
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iters):
        if _time_left() < 10:
            break
        
        # Try swapping two points
        i = random.randint(0, 19)
        j = random.randint(0, 19)
        if i == j:
            continue
        
        old_i = list(current[i])
        old_j = list(current[j])
        
        # Generate new point for i
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        new_i = [new_x, new_y]
        
        # Check validity
        if new_i in current[:i] + current[i+1:]:
            continue
        if abs(new_x) > 100 or abs(new_y) > 100:
            continue
        
        candidate = [list(p) for p in current]
        candidate[i] = new_i
        
        if _has_four_collinear_fast(candidate):
            continue
        
        new_score = _score(candidate)
        if new_score < current_score:
            current = candidate
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
        else:
            # Accept with some probability (simulated annealing-like)
            if random.random() < 0.1:
                current = candidate
                current_score = new_score
    
    return best, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points = None
    best_score = 21  # Initialize to worst possible
    
    # Start with parabola (score 20)
    para = _initial_parabola()
    s = _score(para)
    if s < best_score:
        best_score = s
        best_points = [list(p) for p in para]
    
    # Try dense initialization
    if _time_left() > 20:
        dense = _initial_dense()
        if _validate(dense):
            s = _score(dense)
            if s < best_score:
                best_score = s
                best_points = [list(p) for p in dense]
    
    # Local search from best
    if _time_left() > 15 and best_points is not None:
        improved, improved_score = _local_search(best_points)
        if improved is not None and improved_score < best_score:
            best_score = improved_score
            best_points = improved
    
    # Multiple restarts
    restarts = 0
    while _time_left() > 20:
        restarts += 1
        # Random initialization biased towards collinearity
        pts = []
        # Pick a few lines and sample points from them
        num_lines = random.randint(3, 6)
        for _ in range(num_lines):
            # Random line
            x1 = random.randint(-50, 50)
            y1 = random.randint(-50, 50)
            x2 = random.randint(-50, 50)
            y2 = random.randint(-50, 50)
            if x1 == x2 and y1 == y2:
                continue
            # Sample 2-4 points on this line within bounds
            num_pts = random.randint(2, 4)
            for _ in range(num_pts):
                t = random.uniform(-2, 2)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if -100 <= x <= 100 and -100 <= y <= 100:
                    if [x, y] not in pts:
                        pts.append([x, y])
        # Fill remaining with random points
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if [x, y] not in pts:
                pts.append([x, y])
        pts = pts[:20]
        
        if not _validate(pts):
            continue
        
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in pts]
        
        # Local search from this
        if _time_left() > 15:
            improved, improved_score = _local_search(pts)
            if improved is not None and improved_score < best_score:
                best_score = improved_score
                best_points = [list(p) for p in improved]
    
    if best_points is None:
        return _initial_parabola()
    
    return best_points
# EVOLVE-BLOCK-END
