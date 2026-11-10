import itertools
import random
import numpy as np

def _cross(o, a, b):
    """Cross product (a-o) x (b-o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _collinear3(p, q, r):
    return _cross(p, q, r) == 0

def _build_collinear_triples(points):
    """Return list of (i,j,k) indices where points i,j,k are collinear."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if not _collinear3(points[i], points[j], points[k]):
                    continue
                for l in range(k+1, n):
                    if _collinear3(points[i], points[j], points[l]):
                        return True
    return False

def _max_independent_set(points):
    """
    Exact branch and bound for largest subset with no three collinear.
    Vertices: points. Edges: collinear triples. We want max set with no triple fully included.
    This is the complement of minimum vertex cover in a 3-uniform hypergraph.
    We use branch and bound on the hypergraph.
    """
    n = len(points)
    triples = _build_collinear_triples(points)
    
    # If no triples, answer is n
    if not triples:
        return n
    
    # Precompute for each point, the list of triples containing it
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for idx in t:
            point_triples[idx].append(t)
    
    # Branch and bound
    best = [0]
    
    def _search(remaining, current_size, used_triples):
        """
        remaining: set of indices still available
        current_size: number of points chosen so far
        used_triples: set of triple indices that are "broken" (at least one point not in remaining)
        We want to pick points such that no triple is fully included.
        A triple is "active" if all 3 of its points are in remaining.
        """
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best[0]:
            return
        
        # Find active triples (all 3 points in remaining)
        active = [t for t in triples if all(p in remaining for p in t)]
        
        if not active:
            # Can take all remaining
            new_size = current_size + len(remaining)
            if new_size > best[0]:
                best[0] = new_size
            return
        
        # Pick a triple to branch on
        # Heuristic: pick the triple with the most "conflict" or just the first
        t = active[0]
        
        # Branch: for each point in t, we must exclude it (since we can't have all 3)
        # Actually, we need to exclude at least one point from each active triple.
        # Branch on each point in t: exclude that point.
        for p in t:
            new_remaining = remaining - {p}
            _search(new_remaining, current_size, used_triples)
    
    _search(set(range(n)), 0, set())
    return best[0]

def _score(points):
    """Compute the score (largest subset with no three collinear)."""
    if _has_four_collinear(points):
        return 100  # invalid, penalize
    return _max_independent_set(points)

def _generate_base_config():
    """Generate a base configuration with many collinear triples but no four collinear."""
    # Start with a 4x5 grid, then perturb to avoid 4-collinear
    # Actually, let's use a known construction: points on a conic in a finite field-like structure
    # For simplicity, use a grid and remove/shift points
    
    # Let's try: take a 5x4 grid, then shift every other point in each row by a small amount
    # to break 4-collinear while keeping many 3-collinear
    
    points = []
    for i in range(5):
        for j in range(4):
            x = i * 10
            y = j * 10
            # Perturb to avoid 4-collinear: shift based on position
            if (i + j) % 2 == 0:
                x += 1
            points.append([x, y])
    
    # Check if valid
    if not _has_four_collinear(points):
        return points
    
    # If not, try another construction
    # Use points on y = x^2 but with some points moved to create collinear triples
    # Let's try a different base: 20 points from a 6x6 grid subset
    points2 = []
    for i in range(6):
        for j in range(6):
            if len(points2) < 20:
                x = i * 8
                y = j * 8
                points2.append([x, y])
    
    if not _has_four_collinear(points2):
        return points2
    
    # Fallback: parabola (score 20, but valid)
    return [[x, x * x] for x in range(-10, 10)]

def _local_search(points, max_iters=5000):
    """
    Local search to minimize the score.
    Try swapping points or moving points to reduce the maximum independent set.
    """
    points = [p[:] for p in points]
    n = len(points)
    current_score = _score(points)
    
    # Precompute all valid positions
    valid_positions = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            valid_positions.append([x, y])
    
    best_points = [p[:] for p in points]
    best_score = current_score
    
    for it in range(max_iters):
        # Try to improve by moving one point
        improved = False
        for i in range(n):
            old_point = points[i][:]
            # Try a few random new positions
            for _ in range(20):
                new_pos = random.choice(valid_positions)
                # Check if new_pos is already used
                if any(points[j] == new_pos for j in range(n) if j != i):
                    continue
                points[i] = new_pos[:]
                if _has_four_collinear(points):
                    points[i] = old_point[:]
                    continue
                new_score = _score(points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in points]
                    improved = True
                else:
                    points[i] = old_point[:]
        
        if not improved:
            break
    
    return best_points, best_score

def construct():
    """
    Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest subset with no three collinear.
    """
    random.seed(42)
    
    # Generate several base configurations and try local search on each
    candidates = []
    
    # Base 1: grid-based
    base1 = _generate_base_config()
    candidates.append(base1)
    
    # Base 2: random valid configuration
    for trial in range(5):
        points = []
        attempts = 0
        while len(points) < 20 and attempts < 10000:
            attempts += 1
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = [x, y]
            if p in points:
                continue
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
        
        if len(points) == 20 and not _has_four_collinear(points):
            candidates.append(points)
    
    # Also try a structured configuration: points on multiple lines
    # Take 4 lines, 5 points each, but ensure no 4 collinear overall
    # This is tricky; let's just use the bases we have
    
    best_points = None
    best_score = 100
    
    for base in candidates:
        if not base or len(base) != 20:
            continue
        if _has_four_collinear(base):
            continue
        improved_points, score = _local_search(base, max_iters=2000)
        if score < best_score:
            best_score = score
            best_points = improved_points
    
    if best_points is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_points
