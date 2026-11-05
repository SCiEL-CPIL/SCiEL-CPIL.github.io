import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    best_n = float('inf')
    best_edges = []
    
    # Moore bound for k=5, g=9: 1 + 5*(1 + 5 + 25 + 125) = 781
    # Try N values from Moore bound upward
    
    N_values = [781, 790, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]
    
    for trial in range(30):
        if time.time() > deadline - 5:
            break
            
        for N in N_values:
            if time.time() > deadline - 5:
                break
            
            # k*N must be even
            if (k * N) % 2 != 0:
                continue
            
            adj = [set() for _ in range(N)]
            degrees = [0] * N
            edges = []
            
            target_edges = k * N // 2
            
            success = True
            for step in range(target_edges):
                if time.time() > deadline - 2:
                    success = False
                    break
                
                # Find vertex with min degree - use a simple scan but optimized
                min_deg = k
                min_idx = -1
                for i in range(N):
                    d = degrees[i]
                    if d < min_deg:
                        min_deg = d
                        min_idx = i
                
                if min_idx == -1:
                    success = False
                    break
                
                # Get candidates: vertices with degree < k, not already connected
                candidates = []
                adj_min = adj[min_idx]
                for j in range(N):
                    if j != min_idx and degrees[j] < k and j not in adj_min:
                        candidates.append(j)
                
                if not candidates:
                    success = False
                    break
                
                random.shuffle(candidates)
                
                valid_found = False
                for j in candidates:
                    # Check if adding (min_idx, j) creates a short cycle
                    # Need: no path of length < g-1 between min_idx and j
                    # BFS from min_idx up to depth g-2
                    
                    max_depth = g - 2
                    if max_depth <= 0:
                        valid_found = True
                        break
                    
                    visited = [-1] * N
                    visited[min_idx] = 0
                    queue = deque()
                    queue.append((min_idx, 0))
                    found_short = False
                    
                    while queue:
                        u, dist = queue.popleft()
                        if dist >= max_depth:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = dist + 1
                                if v == j:
                                    found_short = True
                                    break
                                if dist + 1 < max_depth:
                                    queue.append((v, dist + 1))
                            if found_short:
                                break
                        if found_short:
                            break
                    
                    if not found_short:
                        adj[min_idx].add(j)
                        adj[j].add(min_idx)
                        degrees[min_idx] += 1
                        degrees[j] += 1
                        edges.append((min_idx, j))
                        valid_found = True
                        break
                
                if not valid_found:
                    success = False
                    break
            
            if success and all(d == k for d in degrees):
                if N < best_n:
                    best_n = N
                    best_edges = edges[:]
                    break  # Found smallest N for this trial, move to next trial
            
            # If we found a good graph, break early
            if best_n <= 800:
                break
        
        if best_n <= 800:
            break

    if best_n == float('inf'):
        # Fallback: LUW construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges_fallback = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_fallback.append((index[vertex], index[other]))
        return len(order), edges_fallback

    return best_n, best_edges
