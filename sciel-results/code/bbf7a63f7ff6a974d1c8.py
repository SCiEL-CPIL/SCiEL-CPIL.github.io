import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Generate points in PG(2, 5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    nx = 1
                    ny = (y * inv_x) % 5
                    nz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    nx = 0
                    ny = 1
                    nz = (z * inv_y) % 5
                else:
                    inv_z = pow(z, -1, 5)
                    nx = 0
                    ny = 0
                    nz = 1
                
                point = (nx, ny, nz)
                if point not in points:
                    points.append(point)
    
    n = len(points)
    edges = []
    
    # Build adjacency based on dot product 0
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dot = (pi[0] * pj[0] + pi[1] * pj[1] + pi[2] * pj[2]) % 5
            if dot == 0:
                edges.append([i, j])
    
    edge_set = set(tuple(e) for e in edges)
    
    # Build adjacency matrix for common neighbor counting
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    # Find the non-edge with minimum common-neighbor count
    best_edge = None
    best_common = n + 1
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                common = int(np.sum(adj[i] & adj[j]))
                if common < best_common:
                    best_common = common
                    best_edge = [i, j]
                    if common == 0:
                        break
        if best_common == 0:
            break
    
    if best_edge:
        edges.append(best_edge)
    
    return edges
