"""Hexagonal construction with deterministic removal and fast local search for 32 points."""
# EVOLVE-BLOCK-START
import time

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice: a^2 + a*b + b^2."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if abs(a+b) <= r:
                pts.append((a, b))
    return pts

def _hex_ring(r):
    """Points at exactly hex distance r."""
    if r == 0:
        return [(0, 0)]
    inner = set(_hexagon_points(r-1)) if r > 1 else set()
    outer = set(_hexagon_points(r))
    return [p for p in outer if p not in inner]

def _neighbor_count(p, point_set):
    """Count how many neighbors of p are in point_set."""
    a, b = p
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    return sum(1 for da, db in neighbors if (a+da, b+db) in point_set)

def _build_hexagonal_32():
    """Build 32 points: hexagon r=3 (37 pts) minus 5 worst ring-3 points."""
    core = _hexagon_points(2)
    ring3 = _hex_ring(3)
    
    core_set = set(core)
    
    # For each ring-3 point, count its neighbors in the full hexagon (core + ring3)
    full_set = core_set | set(ring3)
    
    # Sort ring-3 points by number of neighbors in full set (ascending - remove least connected first)
    ring3_sorted = sorted(ring3, key=lambda p: _neighbor_count(p, full_set))
    
    # Remove the 5 least-connected ring-3 points
    to_remove = set(ring3_sorted[:5])
    
    selected = core + [p for p in ring3 if p not in to_remove]
    
    return selected

def _fast_local_search(points, time_limit=100, start_time=None):
    """Fast local search: try moving each point to nearby candidates only."""
    if start_time is None:
        start_time = time.time()
    
    points = list(points)
    point_set = set(tuple(p) for p in points)
    best_edges = _count_edges(points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Generate nearby candidates: within hex distance 4 of any current point
    candidate_set = set()
    for a, b in points:
        for da in range(-4, 5):
            for db in range(-4, 5):
                if abs(da) <= 4 and abs(db) <= 4 and abs(da+db) <= 4:
                    na, nb = a+da, b+db
                    if abs(na) <= 64 and abs(nb) <= 64:
                        candidate_set.add((na, nb))
    
    candidates = list(candidate_set)
    
    no_improve = 0
    max_no_improve = 30
    
    while no_improve < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        no_improve += 1
        
        for i in range(len(points)):
            if time.time() - start_time > time_limit:
                break
            
            a1, b1 = points[i]
            best_cand = None
            best_delta = 0
            
            for cand in candidates:
                if cand in point_set:
                    continue
                ca, cb = cand
                
                # Check validity: distance >= 1 from all other points
                valid = True
                for j in range(len(points)):
                    if j == i:
                        continue
                    d2 = _dist2(ca, cb, points[j][0], points[j][1])
                    if d2 < 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute edge delta
                old_edges = _neighbor_count((a1, b1), point_set)
                new_edges = _neighbor_count((ca, cb), point_set)
                delta = new_edges - old_edges
                
                if delta > best_delta:
                    best_delta = delta
                    best_cand = cand
            
            if best_cand is not None and best_delta > 0:
                point_set.discard((a1, b1))
                point_set.add(best_cand)
                points[i] = best_cand
                best_edges += best_delta
                improved = True
        
        if not improved:
            break
    
    return points, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Build hexagonal arrangement
    points = _build_hexagonal_32()
    
    # Verify we have exactly 32 distinct points
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    assert len(set(points)) == 32, "Points are not distinct"
    
    # Verify bounds
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    # Count edges
    edges = _count_edges(points)
    
    # Fast local search to improve
    if time.time() < deadline - 15:
        points, edges = _fast_local_search(points, time_limit=100, start_time=start_time)
    
    # Final verification
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    assert len(set(tuple(p) for p in points)) == 32, "Points are not distinct"
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
