"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    N = 32
    MAX_COORD = 64

    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da * da + db * db + da * db

    def count_edges(points):
        """Count pairs at distance exactly 1."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    def hex_cluster(r):
        """Generate hex cluster of radius r centered at origin."""
        pts = set()
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if max(abs(a), abs(b), abs(a + b)) <= r:
                    pts.add((a, b))
        return list(pts)

    # Generate hex cluster of radius 3 (37 points)
    full_hex = hex_cluster(3)
    assert len(full_hex) == 37, f"Expected 37, got {len(full_hex)}"

    # Precompute adjacency in the full hex: which pairs are at unit distance
    n_pts = 37
    adj = [[False] * n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if dist_sq(full_hex[i], full_hex[j]) == 1:
                adj[i][j] = True
                adj[j][i] = True

    # Count edges in full hex
    full_edges = sum(1 for i in range(n_pts) for j in range(i + 1, n_pts) if adj[i][j])

    # For each point, compute its degree in the full hex
    degrees = [sum(adj[i]) for i in range(n_pts)]

    # We need to remove 5 points. The edge count after removal is:
    # full_edges - (edges incident to removed points) + (edges among removed points)
    # An edge is lost if at least one endpoint is removed.
    # Edges lost = edges with at least one endpoint in R
    # = total edges - edges with both endpoints NOT in R
    # = full_edges - edges_in_remaining

    # Equivalently: edges_in_remaining = full_edges - sum(deg[i] for i in R) + edges_within_R
    # where edges_within_R = number of edges with both endpoints in R

    # We want to maximize edges_in_remaining, i.e., minimize (sum(deg[i] for i in R) - edges_within_R)

    # Try all C(37,5) = 435897 combinations
    best_edges = -1
    best_remove = None

    start_time = time.time()
    for R in combinations(range(n_pts), 5):
        if time.time() > deadline - 10:
            break
        # Compute: sum of degrees of removed points minus edges among removed points
        deg_sum = 0
        for i in R:
            deg_sum += degrees[i]
        edges_within = 0
        for idx in range(5):
            for jdx in range(idx + 1, 5):
                if adj[R[idx]][R[jdx]]:
                    edges_within += 1
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
        assert len(points) == 32
        return [[p[0], p[1]] for p in points]

    # Fallback: try random removals
    random.seed(42)
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        remove_set = set(random.sample(range(n_pts), 5))
        points = [full_hex[i] for i in range(n_pts) if i not in remove_set]
        edges = count_edges(points)
        if edges > best_edges:
            best_edges = edges
            best_remove = remove_set
        if best_edges >= 76:
            break

    if best_remove is not None:
        remove_set = set(best_remove)
        points = [full_hex[i] for i in range(n_pts) if i not in remove_set]
        return [[p[0], p[1]] for p in points]

    # Final fallback
    return [[a, b] for a in range(16) for b in range(2)]
# EVOLVE-BLOCK-END
