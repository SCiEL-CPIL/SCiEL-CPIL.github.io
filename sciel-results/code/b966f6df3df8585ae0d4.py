import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero component is 1
                if x != 0:
                    inv_x = pow(x, 4, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, 4, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)
    
    n = len(points)
    if n != 31:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 2. Build adjacency matrix based on polarity B(u,v) = u1*v1 + u2*v2 + u3*v3 = 0
    A = np.zeros((n, n), dtype=np.int64)
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            b_val = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if b_val == 0:
                A[i, j] = 1
                A[j, i] = 1
    
    edge_count = int(A.sum()) // 2
    if edge_count != 90:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 3. Compute A^2 to find best edge to add
    # C4s created by adding non-edge (i,j) depend on the number of common neighbors, which is A^2[i,j]
    A2 = A @ A
    
    # Find non-edge with minimum A2[i,j]
    min_val = float('inf')
    best_i, best_j = -1, -1
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if A[i, j] == 1:
                continue
            val = int(A2[i, j])
            if val < min_val:
                min_val = val
                best_i, best_j = i, j
    
    if best_i == -1:
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # 4. Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                edges.append([i, j])
    
    edges.append([best_i, best_j])
    
    return edges
