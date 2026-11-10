import time
import random
import itertools
from collections import Counter

def _score(points):
    """Compute the maximum multiplicity of a positive doubled area."""
    counts = {}
    n = len(points)
    for i in range(n - 2):
        x1, y1 = points[i]
        for j in range(i + 1, n - 1):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                d = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _score_fast(points, triple_indices):
    """Faster scoring using precomputed triple indices."""
    counts = {}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    for i, j, k in triple_indices:
        d = abs((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i]))
        if d > 0:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return 0
    return max(counts.values())

def _precompute_triples(n=30):
    """Precompute all C(30,3) triple indices."""
    return [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1) for k in range(j + 1, n)]

def _is_valid(pts):
    return all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts) and len(set(pts)) == 30

def _generate_grid(w, h, ox=0, oy=0):
    return [(x + ox, y + oy) for x in range(w) for y in range(h)]

def _build_arithmetic_progression_set(target_area, n=30):
    """
    Build a set where many triangles have doubled area = target_area.
    
    Strategy: Use points on a line (or near a line) with arithmetic spacing.
    If we have points p_0, p_1, ..., p_{n-1} on a line with spacing vector v,
    and a point q off the line at height h (in the determinant sense),
    then triangle (p_i, p_j, q) has doubled area = h * |j - i| (in units of |v|).
    
    More precisely, if p_i = p_0 + i * v and q is fixed, then
    det(p_i, p_j, q) = det(p_0 + i*v, p_0 + j*v, q) = (j-i) * det(v, q - p_0).
    
    So all triangles (p_i, p_j, q) have doubled area = |j-i| * A where A = det(v, q-p_0).
    
    To maximize multiplicity of a single area value, we want many (i,j) pairs with the same |j-i|.
    The number of pairs with |j-i| = d is (n-1-d) for d = 1, ..., n-1.
    Maximum is at d=1: n-1 pairs.
    
    But we also have triangles among the p_i points themselves (all zero area, excluded).
    
    So with 29 points on a line + 1 off-line point, we get (n-1) = 29 triangles of area A,
    plus 29 triangles of area 2A, 28 of area 3A, etc.
    
    The maximum multiplicity would be 29 (for area A).
    
    But we can do better with a 2D structure.
    
    Let's try: points arranged in rows. If we have points (i, j) for i in range(a), j in range(b),
    with a*b = 30, then triangles within the grid have various areas.
    """
    pass  # Placeholder - we'll implement specific constructions below

def _construct_parallelogram_family(base, u, v, n=30):
    """
    Points: base + i*u + j*v for various (i,j).
    Triangles (p, p+u, p+v) all have doubled area = det(u, v).
    If we have a grid of a x b points, there are (a-1)*(b-1) such parallelogram triangles.
    """
    pass

def _try_grid_construction(w, h, ox, oy):
    """Generate a w x h grid and score it."""
    pts = _generate_grid(w, h, ox, oy)
    if not _is_valid(pts):
        return None, 0
    s = _score(pts)
    return pts, s

def _local_search_polish(points, triple_indices, deadline, max_steps=50000):
    """
    Exhaustive local search: try all single-point moves to nearby positions.
    This is the key improvement - instead of random moves, systematically try
    all possible replacements for each point.
    """
    current = points[:]
    current_score = _score_fast(current, triple_indices)
    best = current[:]
    best_score = current_score
    
    all_points_set = set(current)
    
    # Precompute: for each point, what are the candidate replacements?
    # We'll use a bounded neighborhood to keep it tractable.
    
    step = 0
    while time.time() < deadline and step < max_steps:
        improved = False
        for idx in range(30):
            if time.time() >= deadline:
                break
            old_pt = current[idx]
            ox, oy = old_pt
            
            # Try all nearby points (within radius 5) that aren't already used
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                        continue
                    new_pt = (nx, ny)
                    if new_pt in all_points_set:
                        continue
                    
                    # Apply move
                    current[idx] = new_pt
                    all_points_set.discard(old_pt)
                    all_points_set.add(new_pt)
                    
                    new_score = _score_fast(current, triple_indices)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best = current[:]
                    else:
                        # Revert
                        current[idx] = old_pt
                        all_points_set.discard(new_pt)
                        all_points_set.add(old_pt)
        
        step += 1
        if not improved and step > 1:
            # No improvement in a full sweep, stop
            break
    
    return best, best_score

def _random_local_search(points, triple_indices, deadline, max_steps=100000):
    """Random local search with hill climbing."""
    current = points[:]
    current_score = _score_fast(current, triple_indices)
    best = current[:]
    best_score = current_score
    
    all_pts = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    current_set = set(current)
    
    step = 0
    while time.time() < deadline and step < max_steps:
        idx = random.randint(0, 29)
        old_pt = current[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(3):
            new_pt = random.choice(all_pts)
            if new_pt == old_pt or new_pt in current_set:
                continue
            
            current[idx] = new_pt
            current_set.discard(old_pt)
            current_set.add(new_pt)
            
            new_score = _score_fast(current, triple_indices)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = current[:]
                found_improvement = True
                break
            else:
                current[idx] = old_pt
                current_set.discard(new_pt)
                current_set.add(old_pt)
        
        step += 1
        if not found_improvement and step % 5000 == 0:
            # Periodic check
            pass
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    triple_indices = _precompute_triples(30)
    
    # Phase 1: Try structured initial configurations
    candidates = []
    
    # Various grids
    grid_configs = [
        (5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2),
        (1, 30), (30, 1),
    ]
    for w, h in grid_configs:
        for ox in range(-5, 6, 2):
            for oy in range(-5, 6, 2):
                pts = _generate_grid(w, h, ox, oy)
                if _is_valid(pts):
                    candidates.append(pts)
    
    # Arithmetic progression based: points on a line + offset point
    # 29 points on a line with spacing (1,0), 1 point at (0, h)
    for h_val in [1, 2, 3, 5, 7, 10]:
        pts = [(i, 0) for i in range(29)]
        pts.append((0, h_val))
        # Center it
        cx = sum(p[0] for p in pts) // 30
        cy = sum(p[1] for p in pts) // 30
        pts = [(p[0] - cx, p[1] - cy) for p in pts]
        if _is_valid(pts):
            candidates.append(pts)
    
    # Parallelogram grid: points (i*a, j*b) 
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            for w in [5, 6]:
                for h in [6, 5]:
                    if w * h != 30:
                        continue
                    pts = [(i * a, j * b) for i in range(w) for j in range(h)]
                    cx = sum(p[0] for p in pts) // 30
                    cy = sum(p[1] for p in pts) // 30
                    pts = [(p[0] - cx, p[1] - cy) for p in pts]
                    if _is_valid(pts):
                        candidates.append(pts)
    
    # Evaluate all candidates
    best_points = None
    best_score = -1
    for c in candidates:
        if not _is_valid(c):
            continue
        s = _score_fast(c, triple_indices)
        if s > best_score:
            best_score = s
            best_points = c[:]
    
    if best_points is None:
        # Fallback
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score_fast(best_points, triple_indices)
    
    # Phase 2: Local search polish from best candidate
    if time.time() < deadline - 5:
        polished, polished_score = _local_search_polish(
            best_points, triple_indices, deadline - 10, max_steps=20000
        )
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    # Phase 3: Random local search for additional exploration
    if time.time() < deadline - 3:
        random_best, random_score = _random_local_search(
            best_points, triple_indices, deadline - 2, max_steps=200000
        )
        if random_score > best_score:
            best_points = random_best
            best_score = random_score
    
    # Phase 4: Final polish
    if time.time() < deadline - 1:
        polished2, polished_score2 = _local_search_polish(
            best_points, triple_indices, deadline - 0.5, max_steps=5000
        )
        if polished_score2 > best_score:
            best_points = polished2
            best_score = polished_score2
    
    return best_points
