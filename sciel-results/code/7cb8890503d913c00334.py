import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db

def _edge_count(points):
    """Count edges in a set of points."""
    pts_set = set(points)
    cnt = 0
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set and p < q:
                cnt += 1
    return cnt

def _valid(points):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(points) != 32:
        return False
    if len(set(points)) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    pts_set = set(points)
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set:
                # Distance is exactly 1, which is >= 1.
                # We need to ensure NO distance < 1.
                # In this lattice, integer coordinates imply distances are sqrt(k).
                # The smallest non-zero distance is 1.
                # So any distinct integer points on this lattice have distance >= 1.
                pass
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Use only 10s to be safe
    
    # Generate hexagon of radius 3 (37 points)
    # Points (a, b) such that max(|a|, |b|, |a+b|) <= 3 ?
    # In axial coords (q, r), hex radius R is max(|q|, |r|, |q+r|) <= R.
    # Let's verify count for R=3.
    # R=0: 1
    # R=1: 7
    # R=2: 19
    # R=3: 37
    # This matches.
    
    base_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 3:
                base_points.append((a, b))
                
    # We have 37 points. We need to remove 5.
    current_points = set(base_points)
    
    # Greedy removal: remove point with fewest neighbors in current set
    while len(current_points) > 32:
        min_deg = float('inf')
        min_pt = None
        for p in current_points:
            deg = 0
            for n in _neighbors(p):
                if n in current_points:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_pt = p
        current_points.remove(min_pt)
        
    # Convert to list for local search
    best_points = list(current_points)
    best_score = _edge_count(best_points)
    
    # Local search to polish
    # Try swapping a point in the set with a point outside (from the base hexagon or nearby)
    # Candidate pool: Base points + neighbors of base points
    pool = set(base_points)
    for p in base_points:
        for n in _neighbors(p):
            pool.add(n)
            
    random.seed(42)
    iterations = 0
    while time.time() < deadline and iterations < 100000:
        iterations += 1
        # Pick a random point in current set to remove
        p_in = random.choice(best_points)
        # Pick a random point from pool to add
        p_out = random.choice(list(pool))
        
        if p_in == p_out:
            continue
            
        # Check if p_out is already in set
        if p_out in best_points:
            continue
            
        # Check if p_out is valid (distance >= 1 from all others)
        # Since we are on integer lattice, distinct points have dist >= 1.
        # So just check distinctness.
        
        new_set = set(best_points)
        new_set.remove(p_in)
        new_set.add(p_out)
        
        new_score = _edge_count(list(new_set))
        
        if new_score > best_score:
            best_score = new_score
            best_points = list(new_set)
            
    # Final validity check
    if not _valid(best_points):
        # Fallback to seed if something went wrong
        return [[a, b] for a in range(16) for b in range(2)]
        
    return [[a, b] for a, b in best_points]
