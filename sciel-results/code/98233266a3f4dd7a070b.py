"""Polarity graph on PG(2,5) plus optimal extra edge to minimize C4 count."""
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate all 31 points in PG(2,5) as normalized vectors in F_5^3
    # A point is a 1D subspace, represented by its nonzero vector scaled so first nonzero entry is 1
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first nonzero coordinate equal to 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], 3, 5)  # modular inverse in F_5
                vec = [(v * inv) % 5 for v in vec]
                points.append(tuple(vec))
    
    # Remove duplicates (should be 31 unique points)
    points = list(set(points))
    points.sort()
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency using the standard polarity: x . y = 0 in F_5
    # Two points p, q are adjacent if their representative vectors are orthogonal
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            pi = np.array(points[i])
            pj = np.array(points[j])
            dot = (pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
    
    # The polarity graph should have 90 edges
    num_edges = len(edges)
    
    # If we don't have exactly 90 edges, we may need to adjust
    # For now, proceed with whatever we have
    if num_edges != 90:
        # Fallback: use a different construction or adjust
        # For robustness, if we don't get 90, we'll use a greedy approach
        # But first let's check what we got
        pass
    
    # Count current 4-cycles (should be 0 for a C4-free graph)
    def count_c4s(adj_mat, n):
        """Count number of 4-cycles in the graph."""
        A = adj_mat.astype(np.int64)
        # Number of 4-cycles = (1/8) * sum over all i,j of (A^2)_{ij}^2 - degree corrections
        # More precisely: number of 4-cycles = (1/8) * (sum_{i,j} (A^2)_{ij}^2 - sum_i deg(i)^2 - 2*sum_i deg(i))
        # Actually: each 4-cycle contributes 8 to sum_{i,j} (A^2)_{ij}^2
        # But we need to be careful about triangles and 2-cycles
        # For a simple graph without multi-edges:
        # sum_{i,j} (A^2)_{ij}^2 = sum_j deg(j)^2 + 4*(number of paths of length 2 that are not triangles) ... 
        # Let's use a direct approach for n=31
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mat[i, j] == 0:
                    # Count common neighbors
                    common = np.sum(adj_mat[i] & adj_mat[j])
                    if common >= 2:
                        c4_count += common * (common - 1) // 2
        return c4_count
    
    # For each non-edge, compute how many 4-cycles would be created by adding it
    # When adding edge (u,v), new 4-cycles = number of paths u-a-b-v in original graph
    # = sum over a in N(u) of (number of b in N(v) such that (a,b) is an edge)
    # But we also need a != v and b != u, and a != b
    
    best_extra_edge = None
    best_new_c4s = float('inf')
    
    # Precompute neighbor lists
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                neighbors[i].add(j)
    
    # For each potential edge (u,v) that is not already an edge
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                # Count paths u-a-b-v where a in N(u), b in N(v), (a,b) in E
                # a != v (since v not in N(u) as it's a non-edge... wait, v could be in N(u) if adj[u,v]=1, but we're checking adj[u,v]=0)
                # Actually since adj[u,v]=0, v is not in N(u) and u is not in N(v)
                count = 0
                for a in neighbors[u]:
                    for b in neighbors[v]:
                        if a != b and adj[a, b]:
                            count += 1
                # Each path u-a-b-v corresponds to one 4-cycle u-a-b-v-u
                # But we might be double counting? No, each 4-cycle is counted once per edge addition
                # Actually, a 4-cycle u-a-b-v-u is the same as u-v-b-a-u, but since we're fixing the edge (u,v),
                # each distinct 4-cycle containing (u,v) is counted exactly once
                if count < best_new_c4s:
                    best_new_c4s = count
                    best_extra_edge = (u, v)
    
    # Add the best extra edge
    if best_extra_edge is not None:
        u, v = best_extra_edge
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append((u, v))
    
    # Verify we have exactly 91 edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    # Verify all edges are distinct and valid
    edge_set = set()
    for u, v in edges:
        assert 0 <= u < 31 and 0 <= v < 31
        assert u != v
        key = (min(u, v), max(u, v))
        assert key not in edge_set
        edge_set.add(key)
    
    # Convert to list of lists as required
    result = [[u, v] for u, v in edges]
    
    return result
