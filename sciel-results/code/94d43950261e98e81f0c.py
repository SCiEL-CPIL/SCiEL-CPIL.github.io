import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin before 150s kill
    
    # Field GF(5)
    # Points of PG(2, 5)
    # (x, y, z)
    points = []
    # Affine part: z=1
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    # Line at infinity: z=0
    # (1, m, 0) for m in 0..4
    for m in range(5):
        points.append((1, m, 0))
    # (0, 1, 0)
    points.append((0, 1, 0))
    
    n = len(points)
    # n should be 31
    
    # Adjacency matrix
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check orthogonality
            # x_i x_j + y_i y_j + z_i z_j == 0 mod 5
            val = (points[i][0] * points[j][0] + 
                   points[i][1] * points[j][1] + 
                   points[i][2] * points[j][2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    # Should have 90 edges
    # Compute A^3
    A2 = A @ A
    A3 = A2 @ A
    
    # Find non-edge with min (A^3)[i, j]
    min_val = float('inf')
    best_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_pair = (i, j)
                    
    edges.append([best_pair[0], best_pair[1]])
    
    # Try local search to improve
    # Count C4s in current graph
    def count_c4s(adj):
        # Number of C4s = sum over all pairs (i,j) of C(common_neighbors(i,j), 2) / 2
        # Actually: for each pair of vertices, count common neighbors, then C4s = sum C(common, 2) / 2
        # Wait, that's not quite right. A C4 is a cycle of length 4.
        # Each C4 has 4 edges, so if we count for each pair of vertices the number of common neighbors,
        # each C4 contributes 2 to the sum (once for each pair of opposite vertices).
        # So total C4s = (1/2) * sum_{i<j} C(common_neighbors(i,j), 2)
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] * adj[j])
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    current_c4s = count_c4s(A)
    
    # Try to swap edges to reduce C4s
    # We have 91 edges. Try removing one edge and adding another
    # to see if we can reduce C4 count
    
    improved = False
    for _ in range(50):  # Try a few iterations
        if time.time() > deadline:
            break
            
        best_swap = None
        best_c4s = current_c4s
        
        # Try removing each edge and adding a non-edge
        for e_idx in range(len(edges)):
            if time.time() > deadline:
                break
                
            u, v = edges[e_idx]
            
            # Remove edge
            A[u, v] = 0
            A[v, u] = 0
            
            # Recompute A^3 for this state (expensive but let's try)
            # Actually, let's be smarter: just count C4s after removal
            c4s_after_removal = count_c4s(A)
            
            # Now try adding each non-edge
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if A[i, j] == 0 and not (i == u and j == v) and not (i == v and j == u):
                        # Add edge
                        A[i, j] = 1
                        A[j, i] = 1
                        
                        c4s_with_add = count_c4s(A)
                        
                        if c4s_with_add < best_c4s:
                            best_c4s = c4s_with_add
                            best_swap = (e_idx, i, j)
                        
                        # Undo
                        A[i, j] = 0
                        A[j, i] = 0
            
            # Restore edge
            A[u, v] = 1
            A[v, u] = 1
        
        if best_swap is not None:
            e_idx, new_i, new_j = best_swap
            old_u, old_v = edges[e_idx]
            edges[e_idx] = [new_i, new_j]
            A[old_u, old_v] = 0
            A[old_v, old_u] = 0
            A[new_i, new_j] = 1
            A[new_j, new_i] = 1
            current_c4s = best_c4s
            improved = True
        else:
            break
    
    return edges
