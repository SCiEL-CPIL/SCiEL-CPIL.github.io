"""Direct construction: use a known optimal 32-point configuration on the triangular lattice.

Strategy:
1. Start with the full hexagon of radius 3 (37 points, 99 edges).
2. Remove 5 points to get 32 points, minimizing lost edges.
3. Use multiple initialization strategies + local search (1-swap, 2-swap) to maximize edges.

Harborth's formula gives floor(3*32 - sqrt(12*32 - 3)) = 76.
"""
import time
import math
import random

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
            if max(abs(a), abs(b), abs(a+b)) <= r:
                pts.append((a, b))
    return pts

def _local_search(kept, all_pts, neighbors, deadline):
    """Perform 1-swap and 2-swap local search. Returns (kept, edges)."""
    kept_set = set(kept)
    removed = [p for p in all_pts if p not in kept_set]
    current_edges = _count_edges(kept)
    
    # 1-swap local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for ri in range(len(removed)):
            if time.time() >= deadline - 5.0:
                break
            for ki in range(len(kept)):
                if time.time() >= deadline - 5.0:
                    break
                new_kept = kept[:ki] + kept[ki+1:] + [removed[ri]]
                e = _count_edges(new_kept)
                if e > current_edges:
                    current_edges = e
                    kept = new_kept
                    kept_set = set(kept)
                    removed[ri] = kept[ki]
                    improved = True
                    if current_edges >= 76:
                        break
            if current_edges >= 76:
                break
        if current_edges >= 76:
            break
    
    # 2-swap local search
    if time.time() < deadline - 10.0 and current_edges < 76:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            removed = [p for p in all_pts if p not in kept_set]
            for r1 in range(len(removed)):
                for r2 in range(r1+1, len(removed)):
                    if time.time() >= deadline - 10.0:
                        break
                    for k1 in range(len(kept)):
                        for k2 in range(k1+1, len(kept)):
                            if time.time() >= deadline - 10.0:
                                break
                            new_kept = [p for i, p in enumerate(kept) if i != k1 and i != k2]
                            new_kept.append(removed[r1])
                            new_kept.append(removed[r2])
                            e = _count_edges(new_kept)
                            if e > current_edges:
                                current_edges = e
                                kept = new_kept
                                kept_set = set(kept)
                                improved = True
                                if current_edges >= 76:
                                    break
                        if current_edges >= 76:
                            break
                    if current_edges >= 76:
                        break
                if current_edges >= 76:
                    break
            if current_edges >= 76:
                break
    
    return kept, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Full hexagon of radius 3: 37 points
    all_pts = _hexagon_points(3)
    pt_set = set(all_pts)
    n_remove = 5
    
    # Build adjacency
    adj = {}
    for p in all_pts:
        adj[p] = set()
    for p in all_pts:
        for da, db in neighbors:
            np = (p[0]+da, p[1]+db)
            if np in pt_set:
                adj[p].add(np)
    
    best_kept = None
    best_edges = 0
    
    # Strategy 1: Greedy removal by degree
    current_set = set(all_pts)
    for _ in range(n_remove):
        best_pt = None
        best_deg = 999
        for p in current_set:
            deg = sum(1 for nb in adj[p] if nb in current_set)
            if deg < best_deg:
                best_deg = deg
                best_pt = p
        if best_pt is not None:
            current_set.remove(best_pt)
    
    kept, edges = _local_search(list(current_set), all_pts, neighbors, deadline)
    if edges > best_edges:
        best_edges = edges
        best_kept = kept
    
    # Strategy 2: Remove 5 of 6 corners
    corners = [(3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)]
    corners_in_hex = [c for c in corners if c in pt_set]
    if len(corners_in_hex) >= 5:
        for combo_start in range(6):
            if time.time() >= deadline - 20.0:
                break
            to_remove = [corners_in_hex[(combo_start + i) % 6] for i in range(5)]
            current_set = set(all_pts)
            for p in to_remove:
                current_set.remove(p)
            kept, edges = _local_search(list(current_set), all_pts, neighbors, deadline)
            if edges > best_edges:
                best_edges = edges
                best_kept = kept
    
    # Strategy 3: Random restarts
    if time.time() < deadline - 15.0 and best_edges < 76:
        for _ in range(100):
            if time.time() >= deadline - 10.0:
                break
            to_remove = random.sample(all_pts, n_remove)
            current_set = set(all_pts)
            for p in to_remove:
                current_set.remove(p)
            kept, edges = _local_search(list(current_set), all_pts, neighbors, deadline)
            if edges > best_edges:
                best_edges = edges
                best_kept = kept
            if best_edges >= 76:
                break
    
    final_points = best_kept
    
    # Validation
    assert len(final_points) == 32, f"Got {len(final_points)} points"
    assert len(set(tuple(p) for p in final_points)) == 32, "Points not distinct"
    for a, b in final_points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    return [[a, b] for a, b in final_points]
