"""Construct 32 unit pennies on triangular lattice maximizing unit-distance edges."""
import time
import math
import random

# Deadline guard: kill before 150s wall-clock
_DEADLINE = time.time() + 140.0

def construct():
    """Return exactly 32 distinct integer pairs [a,b] with |a|,|b|<=64,
    mutual distance >= 1, maximizing count of pairs at distance exactly 1.
    Target: 76 edges (Harborth's theorem for n=32)."""
    
    n = 32
    target = 76
    
    # Triangular lattice: point (a,b) represents a + b*omega, omega = exp(i*pi/3)
    # Distance between (a1,b1) and (a2,b2):
    # dx = (a1-a2) + (b1-b2)*cos(60) = (a1-a2) + (b1-b2)/2
    # dy = (b1-b2)*sin(60) = (b1-b2)*sqrt(3)/2
    # dist^2 = dx^2 + dy^2 = (a1-a2 + (b1-b2)/2)^2 + 3*(b1-b2)^2/4
    #        = (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    # So dist^2 = da^2 + da*db + db^2 where da=a1-a2, db=b1-b2
    # Unit distance: da^2 + da*db + db^2 = 1
    # This happens when (da,db) is one of: (1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)
    
    def dist2(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1, verify all distances >= 1."""
        cnt = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                d2 = dist2(points[i], points[j])
                if d2 < 1:
                    return -1  # invalid
                if d2 == 1:
                    cnt += 1
        return cnt
    
    # Strategy: Build a hexagonal/spiral cluster of 32 points that achieves 76 edges.
    # A triangular lattice cluster of n points in a hexagonal shape maximizes edges.
    # For n=32, we need a shape close to hexagonal.
    
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # n=32 is between 19 and 37, so we take a hexagon of radius 2 (19 points) 
    # and add 13 more points around it.
    
    # Let's generate a hexagonal cluster centered at origin.
    # Hexagon of radius r has 1 + 3r(r+1) points.
    # r=0: 1, r=1: 7, r=2: 19, r=3: 37
    
    # Generate all points in hexagon of radius 3 (37 points), then select 32
    # that maximize edges.
    
    # Actually, let's try a direct approach: generate candidate points in a 
    # bounded region, then use local search / greedy to find 32 with 76 edges.
    
    # First, try a known good configuration: hexagonal spiral.
    # Generate points in order of increasing distance from center, forming a spiral.
    
    # Points in triangular lattice within hexagon of radius r:
    # (a,b) with a^2 + a*b + b^2 <= r^2 ... no, that's not right for hexagon.
    # Hexagon in axial coords: max(|a|, |b|, |a+b|) <= r  ... 
    # Actually in the basis where omega = exp(i*pi/3), the hexagonal norm is:
    # For point (a,b), the hexagonal distance from origin is max(|a|, |b|, |a+b|)
    # Wait, let me reconsider. In axial coordinates (a,b) for triangular lattice,
    # the six neighbors of (a,b) are:
    # (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
    # The hexagonal distance (number of steps) from (0,0) to (a,b) is:
    # (|a| + |b| + |a+b|) / 2
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
    
    # Generate hexagon of radius 3: all (a,b) with hex_dist(a,b) <= 3
    # That's 1 + 6 + 12 + 18 = 37 points
    hex3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) <= 3:
                hex3.append([a, b])
    
    # Verify count
    assert len(hex3) == 37, f"Expected 37, got {len(hex3)}"
    
    # We need to remove 5 points from the 37-point hexagon to get 32 points
    # that maximize edges. Removing a point of degree d loses d edges.
    # In the 37-point hexagon, total edges can be computed.
    
    # Let's compute edges in the full hexagon
    full_edges = count_edges(hex3)
    # For a hexagon of radius 3 (37 points), the number of unit edges is:
    # Each interior point has 6 neighbors, boundary points have fewer.
    # Let's just compute it.
    
    # Strategy: try removing each subset of 5 points and find the one with max edges.
    # But C(37,5) = 435,897 which is feasible.
    
    # Actually, let's be smarter. Remove points one at a time greedily,
    # or try all single removals, then all pairs, etc.
    
    # Better: try all combinations of 5 points to remove from the 37.
    # C(37,5) = 435897 - feasible within time limit.
    
    from itertools import combinations
    
    best_score = -1
    best_points = None
    
    # For efficiency, precompute adjacency
    point_set = set(tuple(p) for p in hex3)
    adj = {}
    for p in hex3:
        pt = tuple(p)
        adj[pt] = set()
    for p in hex3:
        pt = tuple(p)
        for da, db in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
            nb = (pt[0]+da, pt[1]+db)
            if nb in point_set:
                adj[pt].add(nb)
    
    # Total edges in full hexagon
    total_edges = sum(len(s) for s in adj.values()) // 2
    
    # When we remove a set S of points, the remaining edges = total_edges 
    # - edges incident to removed points + edges among removed points (counted twice)
    # More precisely: edges in remaining = total_edges - (edges with at least one endpoint in S)
    # = total_edges - sum_{p in S} deg(p) + edges_within_S
    # where edges_within_S counts edges where both endpoints are in S (these were subtracted twice)
    
    # To maximize remaining edges, minimize: sum_{p in S} deg(p) - edges_within_S
    # i.e., minimize sum of degrees of removed points minus internal edges of removed set
    
    # Precompute degrees
    deg = {pt: len(s) for pt, s in adj.items()}
    
    # Try all C(37,5) = 435897 combinations
    # For each, compute loss = sum(deg[p] for p in S) - edges_within(S)
    # Minimize loss to maximize remaining edges
    
    # For speed, we can iterate and compute incrementally
    # But 435897 * 5 operations is fine
    
    start_time = time.time()
    
    hex3_tuples = [tuple(p) for p in hex3]
    idx_map = {pt: i for i, pt in enumerate(hex3_tuples)}
    
    best_loss = float('inf')
    best_remove = None
    
    # Iterate over all 5-subsets
    for combo in combinations(range(37), 5):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute loss
        loss = 0
        for i in combo:
            loss += deg[hex3_tuples[i]]
        # Subtract edges within the removed set
        for ii in range(5):
            for jj in range(ii+1, 5):
                pi = hex3_tuples[combo[ii]]
                pj = hex3_tuples[combo[jj]]
                if pj in adj[pi]:
                    loss -= 1
        
        if loss < best_loss:
            best_loss = loss
            best_remove = combo
    
    if best_remove is not None:
        remove_set = set(hex3_tuples[i] for i in best_remove)
        candidate = [list(pt) for pt in hex3_tuples if pt not in remove_set]
        assert len(candidate) == 32
        score = count_edges(candidate)
        
        if score == target:
            # Verify bounds
            for p in candidate:
                assert abs(p[0]) <= 64 and abs(p[1]) <= 64
            return candidate
    
        # If we didn't get 76, try to improve with local search
        best_points = candidate
        best_score = score
    
    # If the hexagon approach didn't yield 76, try other strategies
    if best_score < target:
        # Try a different base shape or local search
        # Local search: start from best_points, try swapping points
        if best_points is None:
            # Fallback: use the hexagon minus 5 corner points
            # Corners of hex3 are at hex_dist=3
            corners = [pt for pt in hex3_tuples if hex_dist(pt[0], pt[1]) == 3]
            # Remove 5 corners
            remove_set = set(corners[:5])
            best_points = [list(pt) for pt in hex3_tuples if pt not in remove_set]
            best_score = count_edges(best_points)
        
        # Local search to improve
        all_candidates = set(hex3_tuples)
        # Add some extra points outside hex3 for more flexibility
        for a in range(-4, 5):
            for b in range(-4, 5):
                if hex_dist(a, b) <= 4 and (a,b) not in all_candidates:
                    all_candidates.add((a,b))
        
        current = [tuple(p) for p in best_points]
        current_set = set(current)
        current_score = best_score
        
        for iteration in range(10000):
            if time.time() > _DEADLINE - 10.0:
                break
            if current_score >= target:
                break
            
            improved = False
            # Try removing one point and adding another
            for i in range(len(current)):
                if time.time() > _DEADLINE - 10.0:
                    break
                removed = current[i]
                new_set = current_set - {removed}
                
                # Try adding each candidate
                for add_pt in all_candidates:
                    if add_pt in new_set:
                        continue
                    if abs(add_pt[0]) > 64 or abs(add_pt[1]) > 64:
                        continue
                    
                    # Check distance >= 1 from all in new_set
                    valid = True
                    for p in new_set:
                        da = add_pt[0] - p[0]
                        db = add_pt[1] - p[1]
                        if da*da + da*db + db*db < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Count edges in new configuration
                    new_edges = 0
                    new_list = list(new_set) + [add_pt]
                    for ii in range(len(new_list)):
                        for jj in range(ii+1, len(new_list)):
                            da = new_list[ii][0] - new_list[jj][0]
                            db = new_list[ii][1] - new_list[jj][1]
                            d2 = da*da + da*db + db*db
                            if d2 == 1:
                                new_edges += 1
                    
                    if new_edges > current_score:
                        current = new_list
                        current_set = set(current)
                        current_score = new_edges
                        improved = True
                        break
                if improved:
                    break
            if not improved and current_score >= target:
                break
    
        if current_score >= target:
            result = [list(pt) for pt in current]
            for p in result:
                assert abs(p[0]) <= 64 and abs(p[1]) <= 64
            return result
    
    # Final fallback: return whatever we have
    if best_points is not None and len(best_points) == 32:
        return best_points
    
    # Absolute fallback: 16x2 strip (original)
    return [[a, b] for a in range(16) for b in range(2)]
