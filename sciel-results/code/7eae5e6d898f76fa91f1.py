"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build PG(2,5) polarity graph
    # Points are 1D subspaces of F_5^3
    q = 5
    field = list(range(q))
    
    # Enumerate all points [x:y:z] with z != 0 first, then [x:y:0] with y != 0, then [1:0:0]
    # Standard: fix representative so that the first non-zero coordinate is 1
    points = []
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero entry 1
                if x != 0:
                    inv_x = pow(x, q-2, q)  # x^{-1} mod q
                    rep = (1, (y*inv_x) % q, (z*inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    rep = (0, 1, (z*inv_y) % q)
                else:
                    rep = (0, 0, 1)
                if rep not in points:
                    points.append(rep)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity: [x:y:z] -> line x*X + y*Y + z*Z = 0
    # Point P=[x:y:z] is absolute if x*x + y*y + z*z = 0 in F_5
    # Two points P=[x1:y1:z1] and Q=[x2:y2:z2] are adjacent if x1*x2 + y1*y2 + z1*z2 = 0
    
    def dot(p1, p2):
        return (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
    
    # Build adjacency
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Count edges
    edge_count = sum(adj[i, j] for i in range(n) for j in range(i+1, n))
    assert edge_count == 90, f"Expected 90 edges, got {edge_count}"
    
    # Count C4s in current graph
    def count_c4s(adj_matrix):
        # Number of C4s = sum over all pairs (u,v) of C(common_neighbors(u,v), 2)
        # but this counts each C4 twice (two diagonals), so divide by 2
        n = adj_matrix.shape[0]
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.sum(adj_matrix[u] & adj_matrix[v])
                count += common * (common - 1) // 2
        return count // 2
    
    base_c4s = count_c4s(adj)
    
    # Find all non-edges and compute how many new C4s each would add
    # Adding edge (u,v) creates new C4s that include this edge
    # A C4 containing edge (u,v) has form u-a-v-b-u where a,b are common neighbors of u and v
    # Wait, more precisely: a C4 containing edge (u,v) is u-a-v-b-u where:
    #   a is adjacent to u, b is adjacent to v, a is adjacent to v, b is adjacent to u
    #   Actually: u-a-v-b-u means edges (u,a), (a,v), (v,b), (b,u)
    #   So a must be common neighbor of u and v, and b must be common neighbor of u and v
    #   But a ≠ b, and we need a ≠ u,v and b ≠ u,v
    #   The number of such C4s is C(common_neighbors(u,v), 2)
    
    # For each non-edge (u,v), the number of new C4s = C(common_neighbors(u,v), 2)
    # But we also need to check: does adding (u,v) create C4s that don't include (u,v)?
    # No - any new C4 must use the new edge.
    
    non_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v] == 0:
                common = np.sum(adj[u] & adj[v])
                new_c4s = common * (common - 1) // 2
                non_edges.append((new_c4s, u, v))
    
    # Sort by number of new C4s
    non_edges.sort()
    
    # Add the edge that creates the fewest new C4s
    if non_edges:
        min_new, u, v = non_edges[0]
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    return edges
# EVOLVE-BLOCK-END
