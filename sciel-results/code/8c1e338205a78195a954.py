import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    n = 31
    points = []
    points.append((1, 0, 0))
    points.append((1, 1, 0))
    points.append((1, 2, 0))
    points.append((1, 3, 0))
    points.append((1, 4, 0))
    points.append((0, 1, 0))
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
            
    A = np.zeros((n, n), dtype=np.int32)
    edges = []
    
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append([i, j])
                
    A3 = np.linalg.matrix_power(A, 3)
    
    min_paths = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                paths = A3[i, j]
                if paths < min_paths:
                    min_paths = paths
                    best_edge = [i, j]
                    
    if best_edge is None:
        best_edge = [0, 1]
        
    edges.append(best_edge)
    
    return edges
