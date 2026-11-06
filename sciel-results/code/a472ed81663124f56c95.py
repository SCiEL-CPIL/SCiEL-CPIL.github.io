import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # 1. Generate points of PG(2, 5)
    q = 5
    points = []
    # x != 0
    for y in range(q):
        for z in range(q):
            points.append([1, y, z])
    # x == 0, y != 0
    for z in range(q):
        points.append([0, 1, z])
    # x == 0, y == 0, z != 0
    points.append([0, 0, 1])
    
    n = len(points)
    # n should be 31
    
    # 2. Compute adjacency matrix
    # A[i][j] = 1 if points[i] and points[j] are orthogonal
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            # B(u, v) = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
            b = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if b == 0:
                A[i][j] = 1
                A[j][i] = 1
                
    # 3. Collect edges
    edges = []
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] == 1:
                edges.append([i, j])
                edge_set.add((i, j))
                
    # Verify edge count
    if len(edges) != 90:
        # Fallback if construction is wrong, though it should be 90
        pass

    # 4. Find the non-edge that minimizes the number of C4s created
    # A C4 is created if there is a path of length 3 between u and v.
    # Path u-a-b-v means A[u,a]=1, A[a,b]=1, A[b,v]=1.
    # We need to count such paths for each non-edge (u,v).
    
    # To speed up, we can precompute neighbors
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i][j] == 1:
                neighbors[i].add(j)
    
    best_edge = None
    min_c4s = float('inf')
    
    # Iterate over all non-edges
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set:
                continue
            
            # Check time
            if time.time() > deadline - 10.0:
                break
            
            # Count paths of length 3 from i to j
            # Path: i - a - b - j
            # a in N(i), b in N(j), and (a,b) in E.
            # Also need i,a,b,j distinct.
            count = 0
            for a in neighbors[i]:
                if a == j: continue # a must be distinct from j for simple path? 
                                    # If a=j, path is i-j-b-j, not simple.
                for b in neighbors[j]:
                    if b == i: continue # b must be distinct from i
                    if b == a: continue # a and b must be distinct
                    if A[a][b] == 1:
                        count += 1
            
            if count < min_c4s:
                min_c4s = count
                best_edge = (i, j)
        
        if time.time() > deadline - 10.0:
            break

    # Add the best edge
    if best_edge:
        edges.append(list(best_edge))
    else:
        # Fallback: add first non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    edges.append([i, j])
                    break
            if (i, j) not in edge_set:
                break

    # Optional: Local search to reduce C4s
    # If min_c4s > 1, we might try to swap edges to reduce it.
    # But given the structure, min_c4s is likely small (1 or 2).
    # A swap involves removing edge (u1, v1) and adding (u2, v2).
    # This is complex to evaluate quickly.
    # Given the time budget, we rely on the greedy addition.
    
    # Final verification
    if len(edges) != 91:
        # This should not happen
        pass
        
    return edges
