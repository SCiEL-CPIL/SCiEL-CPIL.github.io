import time
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    n = 20
    best_edges = 0
    best_graph = []
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Try multiple random orderings
    # We can run this in a loop until time runs out
    attempts = 0
    while _time_left() > 10.0:
        random.seed(attempts)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        adj = [[False] * n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        graph_edges = []
        
        # Fast check for path of length 3
        # We can use a set for neighbors for O(1) lookup
        neighbor_sets = [set() for _ in range(n)]
        
        for u, v in shuffled_edges:
            # Check if adding (u, v) creates a C4
            # A C4 is u-x-y-v.
            # x in N(u), y in N(x), y in N(v), y != u.
            creates_c4 = False
            for x in neighbors[u]:
                # If x == v, continue? No, x is neighbor of u. If x==v, (u,v) already exists? No, we are adding it.
                # But x is in N(u) from previous steps. So x != v.
                for y in neighbors[x]:
                    if y == u:
                        continue
                    if y in neighbor_sets[v]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                # Add edge
                adj[u][v] = True
                adj[v][u] = True
                neighbors[u].append(v)
                neighbors[v].append(u)
                neighbor_sets[u].add(v)
                neighbor_sets[v].add(u)
                graph_edges.append((u, v))
        
        if len(graph_edges) > best_edges:
            best_edges = len(graph_edges)
            best_graph = graph_edges[:]
            
        attempts += 1
        
    # Return the best graph found, colored 0
    result = [[u, v, 0] for u, v in best_graph]
    return result
