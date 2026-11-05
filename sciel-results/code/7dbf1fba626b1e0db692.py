import itertools
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate hexagonal cluster of radius 3
    # Axial coordinates (a, b)
    # Condition: max(|a|, |b|, |a+b|) <= R
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R:
                points.append((a, b))
    
    n = len(points)
    if n != 37:
        # Fallback if generation failed, though it shouldn't
        return [[a, b] for a in range(16) for b in range(2)]
    
    # Precompute edges
    # Distance squared in axial coords: d^2 = a^2 + b^2 + a*b
    # Unit distance means d^2 = 1
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da * da + db * db + da * db
    
    edges = []
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dist_sq(points[i], points[j]) == 1:
                edges.append((i, j))
                adj[i][j] = True
                adj[j][i] = True
                
    total_edges = len(edges)
    # total_edges should be 90 for R=3
    
    # Precompute degrees
    degrees = [0] * n
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
        
    # We need to remove 5 points to get 32.
    # Maximize remaining edges.
    # Remaining edges = Total Edges - Lost Edges
    # Lost Edges = Sum(deg[i] for i in Removed) - Edges(Within Removed)
    
    best_score = -1
    best_removed = None
    
    # Iterate over all combinations of 5 points to remove
    # There are C(37, 5) = 435,897 combinations
    # For each, calculate lost edges.
    
    # Optimization: Precompute a set of edges for fast lookup?
    # Or just use the adj matrix.
    
    for removed in itertools.combinations(range(n), 5):
        # Calculate sum of degrees
        sum_deg = 0
        for idx in removed:
            sum_deg += degrees[idx]
            
        # Calculate edges within removed set
        edges_within = 0
        r_list = list(removed)
        for i in range(5):
            for j in range(i + 1, 5):
                if adj[r_list[i]][r_list[j]]:
                    edges_within += 1
                    
        lost_edges = sum_deg - edges_within
        remaining_edges = total_edges - lost_edges
        
        if remaining_edges > best_score:
            best_score = remaining_edges
            best_removed = removed
            
        # Check time
        if time.time() > deadline - 10:
            break

    # If we found a solution with 76 edges, it's optimal.
    # If not, we return the best we found.
    
    if best_removed is None:
        # Fallback
        return [[a, b] for a in range(16) for b in range(2)]
        
    removed_set = set(best_removed)
    result = []
    for i in range(n):
        if i not in removed_set:
            result.append([points[i][0], points[i][1]])
            
    # Verify count
    if len(result) != 32:
        return [[a, b] for a in range(16) for b in range(2)]
        
    return result

# EVOLVE-BLOCK-END
