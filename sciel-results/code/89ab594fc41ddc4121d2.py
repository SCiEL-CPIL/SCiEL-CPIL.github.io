import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    # Pre-computed optimal configuration for n=32
    # This is the hexagonal cluster of radius 3 (37 points) minus 5 boundary points
    # forming a path to minimize edge loss.
    
    # Generate the 37 points of the hexagonal cluster of radius 3
    full_hex = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                full_hex.append((a, b))
    
    # The 5 points to remove are a consecutive path on the boundary
    # (3,0), (3,-1), (3,-2), (3,-3), (2,-3)
    remove_set = {
        (3, 0),
        (3, -1),
        (3, -2),
        (3, -3),
        (2, -3)
    }
    
    points = [p for p in full_hex if p not in remove_set]
    
    # Verify count
    if len(points) != 32:
        # Fallback to search if hardcoded set is invalid (should not happen)
        return _fallback_search()
        
    return [[p[0], p[1]] for p in points]

def _fallback_search():
    # This is the original search logic, kept as a safety net
    deadline = time.time() + 140
    N = 32
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db

    def hex_cluster(r):
        pts = set()
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if max(abs(a), abs(b), abs(a + b)) <= r:
                    pts.add((a, b))
        return list(pts)

    full_hex = hex_cluster(3)
    n_pts = 37
    adj = [[False] * n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if dist_sq(full_hex[i], full_hex[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True

    degrees = [sum(adj[i]) for i in range(n_pts)]
    full_edges = sum(1 for i in range(n_pts) for j in range(i + 1, n_pts) if adj[i][j])

    best_edges = -1
    best_remove = None

    for R in combinations(range(n_pts), 5):
        if time.time() > deadline - 10:
            break
        deg_sum = sum(degrees[i] for i in R)
        edges_within = sum(1 for idx in range(5) for jdx in range(idx + 1, 5) if adj[R[idx]][R[jdx]])
        lost = deg_sum - edges_within
        remaining_edges = full_edges - lost
        if remaining_edges > best_edges:
            best_edges = remaining_edges
            best_remove = R
        if best_edges >= 76:
            break

    if best_remove is not None:
        remove_set = set(best_remove)
        points = [full_hex[i] for i in range(n_pts) if i not in remove_set]
        return [[p[0], p[1]] for p in points]
        
    return [[a, b] for a in range(16) for b in range(2)]
# EVOLVE-BLOCK-END
