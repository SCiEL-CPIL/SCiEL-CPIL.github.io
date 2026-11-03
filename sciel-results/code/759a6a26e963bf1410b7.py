import time
import sys
from collections import deque
import math

def construct(k, g):
    t0 = time.time()
    deadline = 115.0

    # Try Generalized Petersen Graphs GP(n, m)
    # We want girth >= g.
    # Candidate n must be >= g.
    
    max_n = 2000
    for n in range(g, max_n + 1):
        if time.time() - t0 > deadline - 5.0:
            break
            
        for m in range(1, n):
            if math.gcd(n, m) != 1:
                continue
                
            # Preliminary check
            valid = True
            limit = g - 2
            for j in range(1, limit):
                val = (j * m) % n
                if val == 1 or val == n - 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Construct and Verify
            num_vertices = 2 * n
            edges = []
            
            # Outer
            for i in range(n):
                edges.append((i, (i + 1) % n))
            # Inner
            for i in range(n):
                edges.append((n + i, n + (i + m) % n))
            # Spokes
            for i in range(n):
                edges.append((i, n + i))
                
            # Build Adj
            adj = [[] for _ in range(num_vertices)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Verify Girth
            min_girth = g
            max_dist = (min_girth - 1) // 2
            is_valid = True
            for start in range(num_vertices):
                dist = [-1] * num_vertices
                parent = [-1] * num_vertices
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= max_dist:
                        continue
                    pu = parent[u]
                    for v in adj[u]:
                        if v == pu:
                            continue
                        dv = dist[v]
                        if dv == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            queue.append(v)
                        elif dv >= du:
                            cycle_len = du + dv + 1
                            if cycle_len < min_girth:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                if not is_valid:
                    break
                    
            if is_valid:
                return num_vertices, edges

    # Fallback: Return a small valid graph if possible?
    # For k=3, g=16, if no GP found, return 0, []
    return 0, []
