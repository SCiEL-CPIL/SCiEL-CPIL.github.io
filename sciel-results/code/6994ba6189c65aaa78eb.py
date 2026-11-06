"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    def dist_squared(a1, b1, a2, b2):
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db

    def count_unit_edges(points):
        count = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if dist_squared(points[i][0], points[i][1], points[j][0], points[j][1]) == 1:
                    count += 1
        return count

    # Generate a compact cluster using a greedy center-out approach
    # We want 32 points
    N = 32
    points = [(0, 0)]
    point_set = {(0, 0)}
    
    # Pre-compute neighbors for a given point to check if they are in the set
    # Neighbors of (a, b) are:
    # (a+1, b), (a-1, b)
    # (a, b+1), (a, b-1)
    # (a+1, b-1), (a-1, b+1)
    # Wait, let's verify the distance formula: d^2 = da^2 + da*db + db^2
    # (1,0): 1+0+0=1. OK.
    # (0,1): 0+0+1=1. OK.
    # (1,-1): 1 + (-1) + 1 = 1. OK.
    # (-1,1): 1 + (-1) + 1 = 1. OK.
    # (-1,0): 1+0+0=1. OK.
    # (0,-1): 0+0+1=1. OK.
    # So neighbors are (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
    
    neighbors = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1)
    ]

    # We will greedily add points.
    # To ensure we don't miss good spots, we can maintain a priority queue or just scan a growing box.
    # Since N=32 is small, we can just scan a box around the current points.
    # The max coordinate will be around 3-4.
    
    while len(points) < N:
        best_candidate = None
        best_score = -1
        best_dist_origin = float('inf')
        
        # Determine search bounds
        # Max coordinate in current points
        max_a = max(abs(p[0]) for p in points)
        max_b = max(abs(p[1]) for p in points)
        
        # Search in a box slightly larger than current extent
        # We expect the cluster to grow roughly in a hexagon shape
        # Radius ~ sqrt(32/3) ~ 3. So bounds -4 to 4 should be safe.
        search_min = -5
        search_max = 5
        
        # Optimization: Only consider candidates that are adjacent to at least one existing point?
        # No, greedy might need to "jump" if the immediate neighbors are blocked, but for a convex shape like a hexagon,
        # the next point is always adjacent to the existing set.
        # So we can just look at the "frontier": neighbors of all existing points that are not in the set.
        
        candidates = set()
        for p in points:
            a, b = p
            for na, nb in neighbors:
                cand = (a + na, b + nb)
                if cand not in point_set:
                    # Check bounds
                    if abs(cand[0]) <= 64 and abs(cand[1]) <= 64:
                        candidates.add(cand)
        
        # Evaluate each candidate
        for cand in candidates:
            a, b = cand
            # Count neighbors in point_set
            edge_count = 0
            for na, nb in neighbors:
                if (a + na, b + nb) in point_set:
                    edge_count += 1
            
            # Tie-breaker: distance from origin (to keep compact)
            dist_origin = a*a + a*b + b*b
            
            if edge_count > best_score or (edge_count == best_score and dist_origin < best_dist_origin):
                best_score = edge_count
                best_dist_origin = dist_origin
                best_candidate = cand
        
        if best_candidate is None:
            # Should not happen for N=32 within bounds
            break
        
        points.append(best_candidate)
        point_set.add(best_candidate)
        
        if time.time() > deadline - 5:
            break

    # Verify and return
    # The greedy method should produce a near-optimal or optimal compact hexagon.
    # For N=32, the optimal is 76 edges.
    # Let's do a quick local search to ensure we hit 76 if greedy missed it.
    
    best_points = points[:]
    best_score = count_unit_edges(points)
    
    # Local search: try to swap a point with a nearby empty spot to improve edge count
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(len(points)):
            a_i, b_i = points[i]
            # Try moving point i to a nearby empty spot
            # Search range: 2 steps away
            for da in range(-2, 3):
                for db in range(-2, 3):
                    if da == 0 and db == 0:
                        continue
                    a_new = a_i + da
                    b_new = b_i + db
                    if abs(a_new) > 64 or abs(b_new) > 64:
                        continue
                    if (a_new, b_new) in point_set:
                        continue
                    
                    # Move
                    point_set.remove((a_i, b_i))
                    point_set.add((a_new, b_new))
                    points[i] = (a_new, b_new)
                    
                    new_score = count_unit_edges(points)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = points[:]
                        improved = True
                    else:
                        # Revert
                        point_set.remove((a_new, b_new))
                        point_set.add((a_i, b_i))
                        points[i] = (a_i, b_i)
                        
                        if time.time() > deadline - 5:
                            break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break

    result = [[p[0], p[1]] for p in best_points]
    return result
# EVOLVE-BLOCK-END
