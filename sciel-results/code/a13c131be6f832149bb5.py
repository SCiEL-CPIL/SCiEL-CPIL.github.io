"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

_NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

def _dist_sq(p1, p2):
    """Squared distance in triangular lattice between two points."""
    da = p1[0] - p2[0]
    db = p1[1] - p2[1]
    return da * da + da * db + db * db

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        for da, db in _NEIGHBORS:
            if (a + da, b + db) in pts:
                count += 1
    return count // 2

def _hex_spiral_points(n=32):
    """
    Generate n points in hexagonal spiral order on the triangular lattice.
    """
    max_coord = 6
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            dist_sq = a * a + a * b + b * b
            x = a + b * 0.5
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x) if (x != 0 or y != 0) else 0.0
            candidates.append((dist_sq, angle, a, b))
    candidates.sort(key=lambda c: (c[0], c[1]))
    result = [(a, b) for _, _, a, b in candidates[:n]]
    return result

def _greedy_construct(n_points=32, seed=42, max_coord=15):
    """
    Greedy construction: build the set one point at a time, always choosing
    the next point that maximizes new edges while keeping the cluster compact.
    This avoids the pitfalls of local search starting from a suboptimal seed.
    """
    rng = random.Random(seed)
    
    # Generate candidate pool
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            candidates.append((a, b))
    
    placed = []
    placed_set = set()
    
    # Start with origin
    placed.append((0, 0))
    placed_set.add((0, 0))
    
    for _ in range(n_points - 1):
        best_point = None
        best_score = None
        best_new_edges = -1
        best_dist_center = float('inf')
        
        # Compute centroid for compactness preference
        if placed:
            cx = sum(p[0] for p in placed) / len(placed)
            cy = sum(p[1] for p in placed) / len(placed)
        else:
            cx, cy = 0, 0
        
        for pt in candidates:
            if pt in placed_set:
                continue
            
            # Check minimum distance constraint
            valid = True
            new_edges = 0
            for pa, pb in placed:
                d = _dist_sq(pt, (pa, pb))
                if d < 1:
                    valid = False
                    break
                if d == 1:
                    new_edges += 1
            
            if not valid:
                continue
            
            # Distance to centroid (approximate)
            dx = pt[0] - cx
            dy = pt[1] - cy
            dist_to_center = abs(dx) + abs(dy) + abs(dx + dy)
            
            # Score: primarily new edges, secondary: closer to center
            if new_edges > best_new_edges or (new_edges == best_new_edges and dist_to_center < best_dist_center):
                best_new_edges = new_edges
                best_dist_center = dist_to_center
                best_point = pt
        
        if best_point is None:
            # No valid point found; try to relax or break
            break
        
        placed.append(best_point)
        placed_set.add(best_point)
    
    return placed

def _local_search(points, deadline, max_coord=10):
    """Fast local search using incremental edge count updates."""
    pts = list(points)
    n = len(pts)
    pts_set = set(pts)
    current_count = _count_edges(pts)
    
    # Precompute adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _dist_sq(pts[i], pts[j]) == 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Candidate replacement points
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
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
            for new_pt in candidates:
                if new_pt == old_pt or new_pt in pts_set:
                    continue
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
                if delta > 0:
                    pts_set.remove(old_pt)
                    pts_set.add(new_pt)
                    pts[i] = new_pt
                    current_count += delta
                    # Update adjacency
                    adj[i] = set()
                    for j in range(n):
                        if i == j:
                            continue
                        if _dist_sq(new_pt, pts[j]) == 1:
                            adj[i].add(j)
                            adj[j].add(i)
                    for j in range(n):
                        if i in adj[j] and _dist_sq(new_pt, pts[j]) != 1:
                            adj[j].discard(i)
                        if i != j and _dist_sq(new_pt, pts[j]) == 1 and i not in adj[j]:
                            adj[j].add(i)
                    improved = True
                    break
    return pts, current_count

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_count = -1
    
    # Strategy 1: Hexagonal spiral (known to be optimal for n=32)
    spiral_pts = _hex_spiral_points(32)
    spiral_count = _count_edges(spiral_pts)
    if spiral_count > best_count:
        best_count = spiral_count
        best_points = list(spiral_pts)
    
    # If we already hit the theoretical optimum, return immediately
    if best_count >= 76:
        return [[a, b] for a, b in best_points]
    
    # Strategy 2: Greedy construction with multiple seeds
    for seed in range(10):
        if time.time() > deadline - 5:
            break
        greedy_pts = _greedy_construct(32, seed=seed)
        if len(greedy_pts) < 32:
            continue
        greedy_count = _count_edges(greedy_pts)
        if greedy_count > best_count:
            best_count = greedy_count
            best_points = list(greedy_pts)
        if best_count >= 76:
            break
    
    # Strategy 3: Local search refinement on the best found so far
    if time.time() < deadline - 5 and best_points is not None:
        improved_pts, count = _local_search(best_points, deadline, max_coord=10)
        if count > best_count:
            best_count = count
            best_points = improved_pts
    
    # Fallback: if something went wrong, try a simple greedy
    if best_points is None or best_count < 70:
        rng = random.Random(0)
        candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                candidates.append((a, b))
        rng.shuffle(candidates)
        
        placed = []
        placed_set = set()
        for pt in candidates:
            if len(placed) >= 32:
                break
            valid = True
            for pp in placed:
                if _dist_sq(pt, pp) < 1:
                    valid = False
                    break
            if valid:
                placed.append(pt)
                placed_set.add(pt)
        
        if len(placed) == 32:
            count = _count_edges(placed)
            if count > best_count:
                best_points = placed
                best_count = count
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
