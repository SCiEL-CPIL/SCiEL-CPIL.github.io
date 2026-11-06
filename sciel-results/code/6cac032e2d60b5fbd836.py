"""Optimize 20-point config: minimize largest 3-collinear-free subset, no four collinear."""
import time
import random
import itertools
from math import gcd

_START = None
_DEADLINE = 135.0  # 15s safety margin before 150s kill

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok(margin=2.0):
    return _elapsed() < _DEADLINE - margin

def _line_key(p1, p2):
    """Canonical line key through two points (a,b,c) with gcd-reduced, sign-normalized."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize sign: make first nonzero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dx*(y - y1) - dy*(x - x1) = 0  =>  -dy*x + dx*y = -dy*x1 + dx*y1
    a = -dy
    b = dx
    c = a * x1 + b * y1
    g2 = gcd(abs(a), abs(b))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    # Normalize sign of (a,b)
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)

def _build_line_counts(points):
    """Return dict: line_key -> count of points on that line."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
    return counts

def _no_four_collinear(points):
    """Check no line has 4+ points. Fast: use line counts."""
    counts = _build_line_counts(points)
    for cnt in counts.values():
        if cnt >= 4:
            return False
    return True

def _build_triples(points):
    """Return list of triples (i,j,k) that are collinear."""
    triples = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            for k in range(j + 1, n):
                if _line_key(points[i], points[k]) == key:
                    triples.append((i, j, k))
    return triples

def _max_indep_bnb(points, triples, timeout=5.0):
    """
    Find the largest subset of points with no three collinear.
    Equivalent to: maximum subset S such that no triple (i,j,k) in `triples` is fully contained in S.
    This is the maximum 3-uniform hypergraph independent set.
    Use branch and bound.
    """
    n = len(points)
    # Build adjacency: for each point, which triples contain it
    point_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        point_triples[i].append(idx)
        point_triples[j].append(idx)
        point_triples[k].append(idx)
    
    # Convert to list of frozensets for fast membership
    triple_set = [set(t) for t in triples]
    
    best = [0]
    best_set = [set()]
    
    # Sort points by degree (number of triples they're in) descending for better pruning
    degree = [len(point_triples[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degree[i])
    
    start_time = time.time()
    
    def _dfs(selected, excluded, remaining, current_size):
        if time.time() - start_time > timeout:
            return
        # Prune: even if we add all remaining, can't beat best
        if current_size + len(remaining) <= best[0]:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
                best_set[0] = set(selected)
            return
        
        # Pick a point from remaining
        # Heuristic: pick the point that's in the most triples with other remaining points
        # For simplicity, just pick the first
        p = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include p
        # Check: adding p doesn't create a collinear triple with two already-selected points
        # A triple is violated if all three points are in selected
        # So we need: no triple (p, a, b) where a,b are both in selected
        can_include = True
        for t_idx in point_triples[p]:
            t = triple_set[t_idx]
            if p in t:
                others = [x for x in t if x != p]
                if all(x in selected for x in others):
                    can_include = False
                    break
        
        if can_include:
            new_selected = selected | {p}
            _dfs(new_selected, excluded, rest, current_size + 1)
        
        # Option 2: exclude p
        _dfs(selected, excluded | {p}, rest, current_size)
    
    _dfs(set(), set(), list(order), 0)
    
    return best[0]

def _score(points):
    """Compute the score: largest subset with no three collinear."""
    triples = _build_triples(points)
    return _max_indep_bnb(points, triples, timeout=3.0)

def _random_point():
    return (random.randint(-100, 100), random.randint(-100, 100))

def _is_valid(points):
    """Check: 20 distinct points, |x|,|y| <= 100, no four collinear."""
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if abs(p[0]) > 100 or abs(p[1]) > 100:
            return False
    return _no_four_collinear(points)

def construct():
    global _START
    _START = time.time()
    
    # Initialize with a 5x4 grid
    xs = [-4, -2, 0, 2, 4]
    ys = [-3, -1, 1, 3]
    points = [(x, y) for x in xs for y in ys]
    
    # Ensure no four collinear (grid 5x4: rows have 5, cols have 4 → violates no-four)
    # Need to perturb. Let's use a different init.
    # Use a 5x4 grid but with slight offsets to break collinearities
    # Actually, let's just start with a random-ish config and rely on local search.
    
    # Better init: points on a curve with some perturbation
    # Start with parabola points (no three collinear) then add more to create triples
    # Actually, let's use a structured init that has some collinear triples but no four.
    
    # Init: 20 points in a 5x4 grid, then jitter to break 4-collinear
    points = []
    for i in range(5):
        for j in range(4):
            x = (i - 2) * 10 + random.randint(-2, 2)
            y = (j - 1) * 10 + random.randint(-2, 2)
            points.append((x, y))
    
    # Remove duplicates
    seen = set()
    unique_points = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
    points = unique_points
    
    # If we have fewer than 20, add random points
    while len(points) < 20:
        p = _random_point()
        if p not in seen:
            seen.add(p)
            points.append(p)
    
    # Ensure no four collinear by replacing problematic points
    for _ in range(100):
        if _no_four_collinear(points):
            break
        # Find a line with 4+ points and replace one
        counts = _build_line_counts(points)
        bad_lines = [k for k, v in counts.items() if v >= 4]
        if not bad_lines:
            break
        # Pick a random point on a bad line and replace it
        line = random.choice(bad_lines)
        a, b, c = line
        # Find points on this line
        on_line = []
        for idx, p in enumerate(points):
            if a * p[0] + b * p[1] == c:
                on_line.append(idx)
        if on_line:
            idx = random.choice(on_line)
            # Replace with a random point
            new_p = _random_point()
            attempts = 0
            while new_p in seen and attempts < 100:
                new_p = _random_point()
                attempts += 1
            if new_p not in seen:
                seen.discard(points[idx])
                points[idx] = new_p
                seen.add(new_p)
    
    if not _is_valid(points):
        # Fallback: use parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    best_points = list(points)
    best_score = _score(points)
    
    # Local search: iteratively replace one point to reduce score
    while _deadline_ok():
        improved = False
        # Try multiple random replacements
        for attempt in range(50):
            if not _deadline_ok():
                break
            idx = random.randint(0, 19)
            old_p = points[idx]
            
            # Generate candidate replacement
            new_p = _random_point()
            attempts = 0
            while new_p in seen and attempts < 50:
                new_p = _random_point()
                attempts += 1
            if new_p in seen:
                continue
            
            # Try the replacement
            points[idx] = new_p
            seen.discard(old_p)
            seen.add(new_p)
            
            if not _no_four_collinear(points):
                points[idx] = old_p
                seen.discard(new_p)
                seen.add(old_p)
                continue
            
            sc = _score(points)
            if sc < best_score:
                best_score = sc
                best_points = list(points)
                improved = True
            else:
                # Revert
                points[idx] = old_p
                seen.discard(new_p)
                seen.add(old_p)
        
        if not improved:
            # Stuck: try a more aggressive perturbation
            # Replace 2-3 random points
            for _ in range(3):
                if not _deadline_ok():
                    break
                for idx in random.sample(range(20), 2):
                    old_p = points[idx]
                    new_p = _random_point()
                    attempts = 0
                    while new_p in seen and attempts < 50:
                        new_p = _random_point()
                        attempts += 1
                    if new_p in seen:
                        continue
                    points[idx] = new_p
                    seen.discard(old_p)
                    seen.add(new_p)
            
            if _no_four_collinear(points):
                sc = _score(points)
                if sc < best_score:
                    best_score = sc
                    best_points = list(points)
            else:
                # Revert to best
                points = list(best_points)
                seen = set(best_points)
    
    # Final validation
    if not _is_valid(best_points):
        return [[x, x * x] for x in range(-10, 10)]
    
    return [[p[0], p[1]] for p in best_points]
