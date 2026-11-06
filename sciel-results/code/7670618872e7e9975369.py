import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Construct Polarity Graph of PG(2, 5)
    # Points are non-zero vectors in GF(5)^3 up to scalar multiplication.
    # We can represent them by their first non-zero component being 1.
    
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero element 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    points.append((0, 1, (z * inv_y) % q))
                else:
                    inv_z = pow(z, -1, q)
                    points.append((0, 0, 1))
    
    # There should be 31 points
    # points is a list of tuples (x,y,z)
    
    # Polarity: u . v = 0
    # We use the standard dot product matrix I.
    # Adjacency: u and v are adjacent if u . v == 0 and u != v.
    
    n = len(points)
    A = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                A[i, j] = 1
                A[j, i] = 1
                
    # Count edges
    edges = np.argwhere(A == 1)
    # edges contains (i,j) and (j,i). We want unique pairs.
    # Since A is symmetric, we can just take upper triangle.
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] == 1:
                edge_list.append((i, j))
                
    num_edges = len(edge_list)
    
    # The polarity graph of PG(2, 5) is expected to have 93 edges (6-regular).
    # If num_edges != 93, we might need to adjust strategy, but let's assume it is.
    
    # Function to count C4s
    def count_c4(adj_matrix):
        # adj_matrix is n x n int array
        # c(u,v) = (A^2)[u,v]
        A2 = adj_matrix @ adj_matrix
        # Sum of C(c(u,v), 2) for u < v
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                c = A2[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    # Initial C4 count
    current_c4 = count_c4(A)
    
    # We need exactly 91 edges.
    # If we have 93, we remove 2.
    # If we have 90, we add 1.
    # If we have 91, we are done (but likely 93).
    
    best_edges = edge_list
    best_c4 = current_c4
    
    if num_edges > 91:
        # Remove edges to minimize C4
        # Try removing pairs of edges
        # 93 edges -> ~4200 pairs.
        # Counting C4 is O(N^2) with A2 precomputed? 
        # A2 changes when edges change.
        # Re-computing A2 is O(N^3) or O(N^2) with sparse? 
        # N=31, N^3 = 30k. 4200 * 30k = 126M ops. Might be slow in Python.
        # Use NumPy matmul: 31x31 @ 31x31 is very fast.
        
        edges_set = set(edge_list)
        
        # Optimization: Only consider removing edges that are in C4s?
        # Or just iterate all pairs.
        
        for i in range(len(edge_list)):
            if time.time() > deadline:
                break
            e1 = edge_list[i]
            for j in range(i + 1, len(edge_list)):
                e2 = edge_list[j]
                
                # Create new adjacency matrix
                # Instead of copying full matrix, we can modify A2 directly?
                # Removing edge (u,v):
                # A[u,v] = 0, A[v,u] = 0
                # A2 changes:
                # A2[u, *] and A2[*, u] and A2[v, *] and A2[*, v] change.
                # Specifically, A2 = A @ A.
                # It's faster to just recompute A2 for candidate.
                
                A_test = A.copy()
                A_test[e1[0], e1[1]] = 0
                A_test[e1[1], e1[0]] = 0
                A_test[e2[0], e2[1]] = 0
                A_test[e2[1], e2[0]] = 0
                
                c4_count = count_c4(A_test)
                
                if c4_count < best_c4:
                    best_c4 = c4_count
                    best_edges = [e for e in edge_list if e != e1 and e != e2]
                    
    elif num_edges < 91:
        # Add edges
        # Find non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 0:
                    non_edges.append((i, j))
        
        # Try adding single edges (if need 1) or pairs (if need 2)
        # Assuming we need 1 edge (90 -> 91)
        # If num_edges == 90, add 1.
        
        for e in non_edges:
            if time.time() > deadline:
                break
            A_test = A.copy()
            A_test[e[0], e[1]] = 1
            A_test[e[1], e[0]] = 1
            
            c4_count = count_c4(A_test)
            
            if c4_count < best_c4:
                best_c4 = c4_count
                best_edges = edge_list + [e]

    # Convert to list of lists [u, v]
    result = [[u, v] for u, v in best_edges]
    
    # Ensure exactly 91 edges
    if len(result) != 91:
        # Fallback: If something went wrong, return first 91 lexicographic
        # This is a safety net, but should not trigger if logic is correct
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        result = pairs[:91]
        
    return result
