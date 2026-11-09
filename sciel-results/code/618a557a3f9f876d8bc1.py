"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

_DEADLINE = 150.0  # evaluator kills at 150s
_SAFETY_MARGIN = 5.0  # stop searching 5s before the limit


def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line in form: dy*x - dx*y = c
    c = dy * x1 - dx * y1
    return (dx, dy, c)


def _build_line_map(points):
    """Build a map from line key -> set of point indices on that line."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return line_map


def _has_four_collinear(points):
    """Check if any four points are collinear."""
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 4:
            return True
    return False


def _count_collinear_triples(points):
    """Count the number of collinear triples (i, j, k) where i<j<k."""
    n = len(points)
    line_map = _build_line_map(points)
    count = 0
    for key, idx_set in line_map.items():
        k = len(idx_set)
        if k >= 3:
            count += k * (k - 1) * (k - 2) // 6
    return count


def _largest_no_three_collinear(points, time_limit=30.0):
    """
    Find the size of the largest subset of points with no three collinear.
    Uses branch and bound with a time limit.
    """
    n = len(points)
    if n <= 2:
        return n
    
    # Precompute for each pair of points, the set of other points on the same line
    # line_through[i][j] = set of k (k != i, j) such that i, j, k are collinear
    line_through = {}
    line_map = _build_line_map(points)
    for key, idx_set in line_map.items():
        idx_list = sorted(idx_set)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a, b = idx_list[i], idx_list[j]
                others = set(idx_list) - {a, b}
                if a not in line_through:
                    line_through[a] = {}
                if b not in line_through[a]:
                    line_through[a][b] = set()
                line_through[a][b].update(others)
                # Also store symmetrically
                if b not in line_through:
                    line_through[b] = {}
                if a not in line_through[b]:
                    line_through[b][a] = set()
                line_through[b][a].update(others)
    
    # For each point, precompute the set of points that form a collinear triple with it and another point
    # Actually, let's use a different approach:
    # For a candidate subset S, S has no three collinear iff for every line, |S ∩ line| <= 2.
    
    # We'll use a greedy + local search approach for the subset, which is faster.
    # Actually, let's do a proper branch and bound but with aggressive pruning.
    
    # Alternative: use the fact that we want the LARGEST subset with no three collinear.
    # This is equivalent to: find the minimum number of points to remove so that no line has 3+ points.
    # Let's just do a simple greedy: sort points, add each if it doesn't create a triple.
    # But we want the EXACT value. Let's do branch and bound with memoization on remaining points.
    
    # For n=20, exact B&B might be too slow. Let's use a heuristic that's good enough.
    # The score is the largest subset with no three collinear.
    
    # Greedy approach: repeatedly pick a point that is in the fewest "conflicting" lines.
    # Actually, let's just compute it exactly for small n, and use heuristic for larger.
    
    # For n=20, let's try a randomized approach:
    # 1. Try all subsets of size k from largest to smallest until we find one with no three collinear.
    # But that's exponential.
    
    # Better: use the following insight. If we have a set of points, the maximum no-three-collinear subset
    # can be found by: for each line with k points, we can keep at most 2. 
    # This is a hitting set / packing problem.
    
    # Let's use a simple greedy with randomization:
    def _greedy_no_three(points, rng, max_iters=1000):
        best_size = 0
        best_subset = set()
        for _ in range(max_iters):
            order = list(range(len(points)))
            rng.shuffle(order)
            selected = set()
            for idx in order:
                # Check if adding idx creates a collinear triple
                ok = True
                for i in selected:
                    for j in selected:
                        if i == j:
                            continue
                    # This is O(k^2) per insertion, O(n^3) total
                # Let's do it properly
                ok = True
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    for j in range(i + 1, len(sel_list)):
                        a, b = sel_list[i], sel_list[j]
                        # Check if idx is on the line through a, b
                        key = _line_key(points[a], points[b])
                        if key is None:
                            continue
                        x1, y1 = points[a]
                        x2, y2 = points[b]
                        x, y = points[idx]
                        # Cross product
                        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                        if cross == 0:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    selected.add(idx)
            if len(selected) > best_size:
                best_size = len(selected)
                best_subset = set(selected)
        return best_size
    
    rng = random.Random(42)
    result = _greedy_no_three(points, rng, max_iters=200)
    return result


def _score(points):
    """Compute the score: largest subset with no three collinear."""
    # Quick check: if no four collinear, and we can find a large no-three subset
    return _largest_no_three_collinear(points)


def _initial_points():
    """Generate an initial set of 20 points with no four collinear."""
    # Start with a parabola, then perturb to break any accidental four-collinear
    points = []
    used = set()
    for x in range(-10, 10):
        y = x * x
        if (x, y) not in used:
            points.append([x, y])
            used.add((x, y))
    # Check for four collinear
    if not _has_four_collinear(points):
        return points
    # Perturb some points
    for _ in range(1000):
        points2 = [p[:] for p in points]
        idx = random.randint(0, 19)
        points2[idx][1] += random.choice([-1, 1])
        if abs(points2[idx][1]) > 100:
            continue
        if tuple(points2[idx]) in used:
            continue
        if not _has_four_collinear(points2):
            return points2
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    # Start with a good initial configuration
    points = _initial_points()
    if _has_four_collinear(points):
        # Fallback: random search for a valid configuration
        for _ in range(10000):
            pts = []
            used = set()
            for _ in range(20):
                for _attempt in range(100):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
                        break
            if len(pts) == 20 and not _has_four_collinear(pts):
                points = pts
                break
    
    best_score = _score(points)
    best_points = [p[:] for p in points]
    
    # Simulated annealing to reduce the score
    # Objective: minimize the largest no-three-collinear subset
    # We'll use moves: pick a random point, move it to a random nearby position
    # Constraint: no four collinear
    
    current_score = best_score
    current_points = [p[:] for p in points]
    
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate a neighbor
        new_points = [p[:] for p in current_points]
        idx = random.randint(0, 19)
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        new_points[idx][0] += dx
        new_points[idx][1] += dy
        
        # Check bounds
        if abs(new_points[idx][0]) > 100 or abs(new_points[idx][1]) > 100:
            continue
        
        # Check distinctness
        seen = set()
        valid = True
        for p in new_points:
            if tuple(p) in seen:
                valid = False
                break
            seen.add(tuple(p))
        if not valid:
            continue
        
        # Check no four collinear
        if _has_four_collinear(new_points):
            continue
        
        # Compute new score
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score < current_score or random.random() < (1.0 / (1.0 + temp)) * (current_score - new_score + 1):
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_points

# EVOLVE-BLOCK-END
