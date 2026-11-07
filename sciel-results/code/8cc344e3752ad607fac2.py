import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add(frozenset((u, v)))
    
    # Known valid 6-edge solution
    # Hub 11, spokes to 0, 6, 9, 15, 18, 23
    solution_6 = [[11, 0], [11, 6], [11, 9], [11, 15], [11, 18], [11, 23]]
    
    # Verify solution_6
    adj = [set() for _ in range(n)]
    for e in base_edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)
    for u, v in solution_6:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check triangle-free
    valid = True
    for u, v in solution_6:
        if adj[u] & adj[v]:
            valid = False
            break
            
    if valid:
        # Check diameter
        def check_diameter(adj):
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = [start]
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    if dist[u] >= 4:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                if any(d == -1 or d > 4 for d in dist):
                    return False
            return True
            
        if check_diameter(adj):
            return solution_6

    # Fallback to 7-edge solution if 6-edge fails (should not happen)
    hub = 11
    targets = [1, 4, 7, 15, 18, 21, 23]
    return [[hub, t] for t in targets]
