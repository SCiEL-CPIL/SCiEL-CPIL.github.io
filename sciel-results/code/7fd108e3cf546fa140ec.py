import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    q = 5
    # Generate points of PG(2, q)
    # Points are equivalence classes of non-zero vectors (x, y, z) in F_q^3
    # We normalize such that the first non-zero component is 1.
    points = []
    seen = set()
    
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                vec = [x, y, z]
                # Find first non-zero
                for i in range(3):
                    if vec[i] != 0:
                        # Inverse in F_q
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    if n != 31:
        # Fallback if generation failed (should not happen)
        # Generate random graph? No, must be valid.
        raise ValueError("Failed to generate PG(2,5) points")

    # Polarity matrix M
    # For PG(2, q) with q odd, a standard polarity is given by x^2 + y^2 + z^2 = 0?
    # Or the symplectic form?
    # The seed code used M = [[1, 0, 0], [0, 1, 0], [0, 0, 4]]
    # This corresponds to x^2 + y^2 + 4z^2 = 0 mod 5.
    # Since 4 = -1 mod 5, this is x^2 + y^2 - z^2 = 0.
    # This is a non-degenerate quadratic form.
    
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 4]], dtype=int)
    
    adj = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(points[i], dtype=int)
            y = np.array(points[j], dtype=int)
            # Check if orthogonal: x^T M y == 0
            val = (x @ M @ y) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    if len(edges) != 90:
        # If the construction yields a different number of edges, 
        # the polarity might be different or normalization issue.
        # For q=5, it should be 90.
        pass

    # Find the best edge to add
    best_edge = None
    min_c4 = float('inf')
    
    # Precompute neighbor lists for speed
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    for i in range(n):
        if time.time() > deadline - 5.0:
            break
        ni = neighbors[i]
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                nj = neighbors[j]
                
                # Count C4s created by adding (i, j)
                # A C4 is i - a - b - j - i
                # a in N(i), b in N(j), a ~ b, a != b
                # Count edges between N(i) and N(j)
                # Subtract common neighbors (where a=b)
                
                if len(ni) == 0 or len(nj) == 0:
                    c4_count = 0
                else:
                    # Count edges between ni and nj
                    # adj[np.ix_(ni, nj)] gives a matrix of 1s where edges exist
                    # Sum of this matrix is the number of edges between the sets
                    # However, this counts (a,b) and (b,a) if we were not careful?
                    # adj is symmetric. np.ix_(ni, nj) extracts the submatrix.
                    # The sum is the number of pairs (a,b) with a in ni, b in nj, and a~b.
                    # This includes cases where a=b if a is in both ni and nj.
                    # We need to exclude a=b.
                    
                    # Number of common neighbors
                    common = np.intersect1d(ni, nj)
                    c_common = len(common)
                    
                    # Total edges between the sets
                    # Note: If a=b is a common neighbor, it contributes 1 to the sum.
                    # We want to exclude these.
                    # So c4_count = (sum of adj[np.ix_(ni, nj)]) - c_common
                    
                    # Optimization: 
                    # Instead of creating the submatrix, we can iterate
                    # But n=31, ni, nj are small.
                    sub_adj = adj[np.ix_(ni, nj)]
                    total_edges = int(np.sum(sub_adj))
                    c4_count = total_edges - c_common
                
                if c4_count < min_c4:
                    min_c4 = c4_count
                    best_edge = [i, j]
    
    if best_edge:
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append([u, v])
    else:
        # Should not happen
        pass

    # Verify
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    assert len(set(tuple(sorted(e)) for e in edges)) == 91, "Duplicate edges"

    return edges
