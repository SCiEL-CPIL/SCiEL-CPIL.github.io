"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    """Canonical key for the line through two points (a, b, c) where ax+by=c, normalized."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    g = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    # Canonical sign: make first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _build_line_map(points):
    """Map line_key -> list of point indices on that line."""
    line_map = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
    # Deduplicate
    for key in line_map:
        line_map[key] = sorted(set(line_map[key]))
    return line_map

def _no_four_collinear(points):
    """Check that no line contains 4 or more points."""
    line_map = _build_line_map(points)
    for key, idxs in line_map.items():
        if len(idxs) >= 4:
            return False
    return True

def _collinear_triples(points):
    """Return list of frozensets of 3 indices that are collinear."""
    line_map = _build_line_map(points)
    triples = []
    for key, idxs in line_map.items():
        if len(idxs) >= 3:
            for combo in combinations(idxs, 3):
                triples.append(frozenset(combo))
    return triples

def _max_independent_set(triples, n):
    """
    Find the maximum subset of {0..n-1} containing no triple from `triples`.
    Uses branch and bound.
    """
    if not triples:
        return n
    
    # Build adjacency: for each vertex, list of triples containing it
    vertex_to_triples = [[] for _ in range(n)]
    for t in triples:
        for v in t:
            vertex_to_triples[v].append(t)
    
    best = [0]
    
    def bnb(remaining, current_size, forbidden):
        """
        remaining: frozenset of vertices still available
        current_size: size of independent set so far
        forbidden: frozenset of vertices that cannot be added (because adding them would complete a triple)
        """
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best[0]:
            return
        
        # Greedy upper bound: try to see how many we can add
        # Simple bound: len(remaining)
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with most constraints
        v = None
        max_conflicts = -1
        for vertex in remaining:
            conflicts = len(forbidden & remaining)
            # Count how many triples would be violated if we add this vertex
            # Actually, let's just pick the first or use a heuristic
            v = vertex
            break
        
        # Better heuristic: pick vertex that appears in most remaining triples
        # But for simplicity, let's try all and use the first
        # Actually, let's pick a vertex and branch include/exclude
        
        # Try including v
        new_forbidden = set(forbidden)
        for t in vertex_to_triples[v]:
            t_set = set(t) - {v}
            if t_set <= remaining:
                new_forbidden |= t_set
        
        new_remaining = remaining - {v} - new_forbidden
        if current_size + 1 + len(new_remaining) > best[0]:
            bnb(frozenset(new_remaining), current_size + 1, frozenset(new_forbidden))
        
        # Try excluding v
        new_remaining = remaining - {v}
        if current_size + len(new_remaining) > best[0]:
            bnb(frozenset(new_remaining), current_size, frozenset(forbidden))
    
    bnb(frozenset(range(n)), 0, frozenset())
    return best[0]

def _score_configuration(points):
    """Compute the maximum independent set size (largest subset with no three collinear)."""
    triples = _collinear_triples(points)
    if not triples:
        return len(points)
    return _max_independent_set(triples, len(points))

def _random_valid_points(n=20, max_coord=100, seed=None):
    """Generate n random distinct integer points with no four collinear."""
    rng = random.Random(seed)
    points = set()
    attempts = 0
    while len(points) < n and attempts < 100000:
        x = rng.randint(-max_coord, max_coord)
        y = rng.randint(-max_coord, max_coord)
        p = (x, y)
        if p not in points:
            points.add(p)
            # Check no four collinear
            pts_list = list(points)
            if _no_four_collinear(pts_list):
                attempts = 0
            else:
                points.remove(p)
                attempts += 1
        else:
            attempts += 1
    return [list(p) for p in points]

def _try_improve(points, triples, current_score, rng, max_iters=5000):
    """
    Try to improve by swapping one point with a random new point.
    Returns (new_points, new_score) if improved, else (None, None).
    """
    n = len(points)
    for _ in range(max_iters):
        # Pick a random index to replace
        idx = rng.randint(0, n - 1)
        # Generate a random new point
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        new_point = (x, y)
        
        # Check it's not already in the set
        if any(p[0] == x and p[1] == y for p in points):
            continue
        
        # Try the swap
        new_points = [list(p) for p in points]
        new_points[idx] = [x, y]
        
        # Check no four collinear
        if not _no_four_collinear(new_points):
            continue
        
        # Compute new score
        new_triples = _collinear_triples(new_points)
        new_score = _max_independent_set(new_triples, n)
        
        if new_score < current_score:
            return new_points, new_score
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    rng = random.Random(42)
    
    # Start with a random valid configuration
    best_points = _random_valid_points(20, 100, seed=42)
    if len(best_points) < 20:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    best_score = _score_configuration(best_points)
    
    # Local search
    while time.time() < deadline:
        triples = _collinear_triples(best_points)
        new_points, new_score = _try_improve(best_points, triples, best_score, rng, max_iters=200)
        
        if new_points is not None:
            best_points = new_points
            best_score = new_score
        
        # Occasionally try a fresh random start
        if rng.random() < 0.1:
            candidate = _random_valid_points(20, 100, seed=rng.randint(0, 1000000))
            if len(candidate) == 20:
                cand_score = _score_configuration(candidate)
                if cand_score < best_score:
                    best_points = candidate
                    best_score = cand_score
    
    return [[p[0], p[1]] for p in best_points]
# EVOLVE-BLOCK-END
