"""Improved program: hexagonal construction with local search for 32 points on triangular lattice."""
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

def _hex_dist(a, b):
    return max(abs(a), abs(b), abs(a+b))

def _build_hexagonal_32():
    """Build 32 points in a hexagonal arrangement centered at origin."""
    core = _hexagon_points(2)
    ring3 = _hex_ring(3)
    
    def neighbor_count_in(p, point_list):
        ps = set(point_list)
        a, b = p
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        return sum(1 for da, db in neighbors if (a+da, b+db) in ps)
    
    ring3_sorted = sorted(ring3, key=lambda p: neighbor_count_in(p, core), reverse=True)
    
    selected = list(core)
    for p in ring3_sorted:
        if len(selected) >= 32:
            break
        selected.append(p)
    
    return selected

def _local_search(points, max_iters=5000, time_limit=None, start_time=None):
    """Hill-climbing local search to maximize edge count via swaps."""
    if time_limit is None:
        time_limit = 110
    if start_time is None:
        start_time = time.time()
    
    points = list(points)
    point_set = set(tuple(p) for p in points)
    best_edges = _count_edges(points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Precompute candidate replacement points (within bounds)
    all_candidates = []
    for a in range(-64, 65):
        for b in range(-64, 65):
            if abs(a+b) <= 64:  # Stay in reasonable hex region
                all_candidates.append((a, b))
    
    no_improve = 0
    iters = 0
    
    while iters < max_iters:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        iters += 1
        
        for i, (a1, b1) in enumerate(points):
            if time.time() - start_time > time_limit:
                break
            
            # Try swapping points[i] with each candidate not in current set
            for cand in all_candidates:
                if cand in point_set:
                    continue
                ca, cb = cand
                
                # Check if swap is valid: new point must not be at distance < 1 from any other
                # i.e., squared distance >= 1 for all other points
                valid = True
                for j, (a2, b2) in enumerate(points):
                    if j == i:
                        continue
                    d2 = _dist2(ca, cb, a2, b2)
                    if d2 < 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute edge count change
                # Remove edges involving old point (a1,b1)
                old_edges = 0
                for da, db in neighbors:
                    if (a1+da, b1+db) in point_set:
                        old_edges += 1
                
                # Add edges involving new point (ca,cb)
                new_edges = 0
                for da, db in neighbors:
                    if (ca+da, cb+db) in point_set:
                        new_edges += 1
                
                # But we need to be careful: the set changes
                # Let's just compute full edge count if delta > 0
                delta = new_edges - old_edges
                if delta > 0:
                    # Perform swap
                    points[i] = cand
                    point_set.discard((a1, b1))
                    point_set.add(cand)
                    best_edges += delta
                    improved = True
                    break
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                break
        else:
            no_improve = 0
    
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
    
    # Count edges for verification
    edges = _count_edges(points)
    
    # Local search to improve
    if time.time() < deadline - 10:
        points, edges = _local_search(points, max_iters=10000, time_limit=100, start_time=start_time)
    
    # Final verification
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    assert len(set(tuple(p) for p in points)) == 32, "Points are not distinct"
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
