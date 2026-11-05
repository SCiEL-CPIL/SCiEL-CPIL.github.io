import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 20s safety margin before 160s limit
    
    N = 32
    MAX_COORD = 64
    
    # The 6 neighbor offsets on triangular lattice
    NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db
    
    def count_edges_fast(points_set):
        """Count edges using set lookup - O(n) instead of O(n^2)."""
        count = 0
        for p in points_set:
            for da, db in NEIGHBORS:
                q = (p[0] + da, p[1] + db)
                if q in points_set:
                    count += 1
        return count // 2  # Each edge counted twice
    
    def is_valid_fast(points_set):
        """Check all pairs have distance >= 1 using set lookup."""
        for p in points_set:
            for da, db in NEIGHBORS:
                q = (p[0] + da, p[1] + db)
                if q in points_set:
                    return False
        return True
    
    def hex_cluster(r):
        """Generate hex cluster of radius r centered at origin."""
        pts = set()
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if max(abs(a), abs(b), abs(a + b)) <= r:
                    pts.add((a, b))
        return list(pts)

    # Generate initial hex cluster of radius 3 (37 points)
    full_hex = hex_cluster(3)
    if len(full_hex) != 37:
        # Fallback if hex generation is flawed
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Try all possible removals of 5 points from the 37-point hex
    # Use greedy approach first, then try to improve
    current_points = full_hex[:]
    current_set = set(current_points)
    
    # Greedy removal of 5 points with lowest degree
    for _ in range(5):
        if len(current_points) <= N:
            break
            
        # Calculate degrees of all points in current_points
        degrees = {p: 0 for p in current_points}
        for p in current_points:
            for da, db in NEIGHBORS:
                q = (p[0] + da, p[1] + db)
                if q in current_set:
                    degrees[p] += 1
        
        # Find the point with minimum degree
        min_deg_point = min(current_points, key=lambda x: degrees[x])
        
        # Remove it
        current_points.remove(min_deg_point)
        current_set.remove(min_deg_point)
        
    # We should have 32 points now
    if len(current_points) != N:
        # If something went wrong, fallback to random removal
        random.seed(42)
        idxs = list(range(37))
        random.shuffle(idxs)
        remove_set = set(idxs[:5])
        current_points = [full_hex[i] for i in range(37) if i not in remove_set]
        current_set = set(current_points)
        
    # Verify validity
    if not is_valid_fast(current_set):
        # Should not happen for distinct lattice points, but just in case
        return [[a, b] for a in range(16) for b in range(2)]
        
    # Count edges
    edges = count_edges_fast(current_set)
    
    # If we didn't reach the theoretical maximum of 76, try local search to improve
    if edges < 76:
        points = current_points[:]
        points_set = set(points)
        best_points = points[:]
        best_edges = edges
        
        # Local search with fast edge counting
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                orig = points[i]
                orig_set = points_set.copy()
                # Try moving point i to a nearby position
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0:
                            continue
                        cand = (orig[0] + da, orig[1] + db)
                        if abs(cand[0]) > MAX_COORD or abs(cand[1]) > MAX_COORD:
                            continue
                        
                        # Check if cand is already occupied
                        if cand in points_set:
                            continue
                            
                        # Check validity: new position must be >=1 from all other points
                        valid = True
                        for da2, db2 in NEIGHBORS:
                            q = (cand[0] + da2, cand[1] + db2)
                            if q in points_set and q != orig:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute new edge count
                        points[i] = cand
                        points_set = set(points)
                        new_edges = count_edges_fast(points_set)
                        
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_points = points[:]
                            improved = True
                            if best_edges >= 76:
                                break
                        else:
                            points[i] = orig
                            points_set = set(points)
                    if best_edges >= 76:
                        break
                if best_edges >= 76:
                    break
            
        if best_edges > edges:
            current_points = best_points
            edges = best_edges
            
    return [[p[0], p[1]] for p in current_points]
# EVOLVE-BLOCK-END
