import math
import itertools

def construct():
    # Generate points in axial coordinates (q, r)
    # Hexagonal distance: (|q| + |r| + |q+r|) // 2
    
    def hex_dist(q, r):
        return (abs(q) + abs(r) + abs(q + r)) // 2

    # Get all points with dist <= 3 (37 points)
    # The optimal 32-point configuration is likely a subset of these.
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if hex_dist(q, r) <= 3:
                points.append((q, r))
                
    n = len(points)
    if n < 32:
        # Fallback if logic changes, but 37 >= 32
        return [[p[0], p[1]] for p in points[:32]]
        
    # Build adjacency list
    # dist_sq == 1 iff adjacent
    def are_adjacent(p1, p2):
        q1, r1 = p1
        q2, r2 = p2
        dq = q1 - q2
        dr = r1 - r2
        return dq*dq + dq*dr + dr*dr == 1

    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if are_adjacent(points[i], points[j]):
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Calculate total edges in the full graph of 37 points
    total_edges = sum(len(neigh) for neigh in neighbors) // 2
    
    # We want to choose 32 points. This is equivalent to removing 5 points.
    # We search for the removal set that maximizes the remaining edges.
    # Max possible edges is 76.
    
    best_edges = -1
    best_removals = None
    
    # Iterate over all combinations of 5 points to remove
    # There are C(37, 5) = 435,897 combinations
    for removals in itertools.combinations(range(n), 5):
        removals_set = set(removals)
        
        # Calculate edges lost
        # Edges lost = (sum of degrees of removed points) - (edges within removed points)
        sum_deg = 0
        for idx in removals:
            sum_deg += len(neighbors[idx])
            
        edges_in_removals = 0
        # Check pairs within removals
        rem_list = list(removals)
        for i in range(len(rem_list)):
            for j in range(i+1, len(rem_list)):
                if rem_list[j] in neighbors[rem_list[i]]:
                    edges_in_removals += 1
                    
        lost_edges = sum_deg - edges_in_removals
        current_edges = total_edges - lost_edges
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_removals = removals
            
            # If we reached the theoretical maximum, we can stop early
            if best_edges >= 76:
                break
                
    # Reconstruct the points
    if best_removals is None:
        # Should not happen
        return [[p[0], p[1]] for p in points[:32]]
        
    removals_set = set(best_removals)
    final_points = [points[i] for i in range(n) if i not in removals_set]
    
    return [[p[0], p[1]] for p in final_points]
