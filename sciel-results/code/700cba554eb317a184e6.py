import numpy as np
import time

def construct():
    n = 31
    # Define points in PG(2, 5)
    # Indices 0-5: z=0
    # Indices 6-30: z=1
    points = []
    
    # z=0 points: (x, y, 0)
    # (1,0), (1,1), (1,2), (1,3), (1,4), (0,1)
    pts_z0 = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
        (1, 3, 0),
        (1, 4, 0),
        (0, 1, 0)
    ]
    
    # z=1 points: (x, y, 1)
    for i in range(5):
        for j in range(5):
            pts_z0.append((i, j, 1))
            
    points = pts_z0
    
    # Build adjacency matrix
    # Bilinear form: u0*v0 + u1*v1 + u2*v2 == 0 mod 5
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if val == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    # Count edges to verify
    edge_count = sum(A.sum(axis=0)) // 2
    # Should be 90
    
    # Compute A^3
    A3 = np.linalg.matrix_power(A, 3)
    
    # Find the non-edge (i, j) with minimum A3[i, j]
    min_val = float('inf')
    best_pair = (-1, -1)
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                if A3[i, j] < min_val:
                    min_val = A3[i, j]
                    best_pair = (i, j)
                    
    # Add the best edge
    u, v = best_pair
    A[u, v] = 1
    A[v, u] = 1
    
    # Function to count C4s
    def count_c4s(adj):
        # A2 = adj @ adj
        # trace(A4) = sum(A2 * A2)
        # sum_d_d_minus_1 = sum(d_i * (d_i - 1))
        # m = number of edges
        A2 = adj @ adj
        tr_A4 = np.sum(A2 * A2)
        degrees = adj.sum(axis=0)
        sum_d_d_minus_1 = np.sum(degrees * (degrees - 1))
        m = np.sum(degrees) // 2
        return (tr_A4 - 2*m - sum_d_d_minus_1) // 8

    current_score = count_c4s(A)
    
    # Local Search
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    # Precompute list of edges and non-edges for faster sampling?
    # Or just sample indices.
    
    # To speed up, we can maintain A2? 
    # Updating A2 for a single edge swap is O(n^2) or O(n).
    # A2[i,j] = sum_k A[i,k]A[k,j].
    # If we add edge (u,v), A2 changes for rows/cols u,v.
    # But for simplicity, n=31 is small. Matrix mult is very fast.
    
    while time.time() - start_time < time_limit:
        # Pick random edge to remove
        # Get list of edges
        # This is O(m) each time. 91 edges. Fast.
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        
        if not edges or not non_edges:
            break
            
        e_out = edges[np.random.randint(len(edges))]
        e_in = non_edges[np.random.randint(len(non_edges))]
        
        u_out, v_out = e_out
        u_in, v_in = e_in
        
        # Apply swap
        A[u_out, v_out] = 0
        A[v_out, u_out] = 0
        A[u_in, v_in] = 1
        A[v_in, u_in] = 1
        
        new_score = count_c4s(A)
        
        # Accept if better
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            A[u_out, v_out] = 1
            A[v_out, u_out] = 1
            A[u_in, v_in] = 0
            A[v_in, u_in] = 0
            
    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
                
    return edges
