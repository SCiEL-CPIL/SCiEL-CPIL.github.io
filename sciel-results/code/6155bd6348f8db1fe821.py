# EVOLVE-BLOCK-START
import math
import time
import random

_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _dist_sq(p1, p2):
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da * da + da * db + db * db

def _count_edges(points):
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in pts:
                count += 1
    return count // 2

def _get_hexagon_points(radius_sq_max=4):
    """Get all points with d^2 <= radius_sq_max."""
    pts = []
    # Radius 2 covers d^2 up to 4.
    # a, b range [-2, 2] is sufficient for d^2 <= 4?
    # Max d^2 for (2,2) is 4+4+4=12.
    # We just iterate a,b in [-3,3] and filter.
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a * a + a * b + b * b <= radius_sq_max:
                pts.append((a, b))
    return pts

def _get_ring_points(radius_sq=7):
    """Get all points with d^2 == radius_sq."""
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if a * a + a * b + b * b == radius_sq:
                pts.append((a, b))
    return pts

def _angle_of_point(p):
    a, b = p
    x = a + b * 0.5
    y = b * math.sqrt(3) / 2.0
    return math.atan2(y, x)

def _construct_optimal_32():
    """
    Construct 32 points: Hexagon (d^2<=4) + 7 points from ring d^2=7.
    Try all 12 rotations of the 7-point arc to maximize edges.
    """
    base_pts = _get_hexagon_points(4)
    ring_pts = _get_ring_points(7)
    
    # Sort ring points by angle
    ring_pts_sorted = sorted(ring_pts, key=_angle_of_point)
    
    best_count = -1
    best_points = None
    
    # The ring has 12 points. We want to pick 7 consecutive ones.
    # There are 12 possible arcs of length 7.
    n_ring = len(ring_pts_sorted)
    if n_ring < 7:
        # Fallback if something is wrong
        return base_pts + ring_pts[:7]
        
    for start in range(n_ring):
        # Get 7 points starting from 'start'
        arc_pts = []
        for i in range(7):
            arc_pts.append(ring_pts_sorted[(start + i) % n_ring])
        
        candidate_pts = base_pts + arc_pts
        
        # Quick check: are they distinct?
        if len(set(candidate_pts)) != 32:
            continue
            
        count = _count_edges(candidate_pts)
        if count > best_count:
            best_count = count
            best_points = candidate_pts
            
    return best_points

def _local_search_fast(points, deadline):
    """
    Fast local search to fix any suboptimal configurations.
    """
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    current_count = _count_edges(pts)
    
    # Precompute adjacency
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _dist_sq(pts[i], pts[j]) == 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Candidate spots: empty spots within distance 2 of any point
    # To be fast, we just generate a small window around the cluster
    min_a = min(p[0] for p in pts) - 2
    max_a = max(p[0] for p in pts) + 2
    min_b = min(p[1] for p in pts) - 2
    max_b = max(p[1] for p in pts) + 2
    
    candidates = []
    for a in range(min_a, max_a + 1):
        for b in range(min_b, max_b + 1):
            if (a, b) not in pts_set:
                candidates.append((a, b))
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old_pt = pts[i]
            edges_lost = len(adj[i])
            
            # Try moving to each candidate
            # Optimization: Only try candidates that are close
            # But for n=32, even 100 candidates * 32 points * 32 checks is fast
            best_delta = 0
            best_new_pt = None
            
            for new_pt in candidates:
                if new_pt == old_pt:
                    continue
                # Check validity: distance >= 1 from all others
                valid = True
                edges_gained = 0
                for j in range(n):
                    if i == j:
                        continue
                    d = _dist_sq(new_pt, pts[j])
                    if d < 1:
                        valid = False
                        break
                    if d == 1:
                        edges_gained += 1
                        
                if not valid:
                    continue
                    
                delta = edges_gained - edges_lost
                if delta > best_delta:
                    best_delta = delta
                    best_new_pt = new_pt
                    
            if best_new_pt is not None and best_delta > 0:
                # Perform move
                pts_set.remove(old_pt)
                pts_set.add(best_new_pt)
                pts[i] = best_new_pt
                current_count += best_delta
                
                # Update adjacency
                # Remove old edges
                for j in adj[i]:
                    adj[j].discard(i)
                adj[i].clear()
                
                # Add new edges
                for j in range(n):
                    if i == j:
                        continue
                    if _dist_sq(best_new_pt, pts[j]) == 1:
                        adj[i].add(j)
                        adj[j].add(i)
                        
                improved = True
                break # Restart scan from 0 after a move
                
    return pts, current_count

def construct():
    deadline = time.time() + 140.0
    
    # 1. Direct optimal construction
    best_points = _construct_optimal_32()
    best_count = _count_edges(best_points)
    
    # 2. If we hit optimum, return
    if best_count >= 76:
        return [[a, b] for a, b in best_points]
        
    # 3. Local search to fix
    if time.time() < deadline - 5:
        improved_pts, count = _local_search_fast(best_points, deadline)
        if count > best_count:
            best_count = count
            best_points = improved_pts
            
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
