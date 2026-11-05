import time
import random

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _check_c4(u, v, neighbors, neighbor_sets):
    # Check if adding (u,v) creates a C4
    # Path u-x-y-v where x in N(u), y in N(x), y in N(v)
    for x in neighbors[u]:
        # x is neighbor of u. We add u-v.
        # If x == v, we already have u-v? No, we are checking before adding.
        # But if x==v, then v is in N(u). We shouldn't add u-v if it exists.
        # The caller ensures u,v not connected.
        for y in neighbors[x]:
            if y == u:
                continue
            if y in neighbor_sets[v]:
                return True
    return False

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_edges = 0
    best_graph = []
    
    attempts = 0
    while _time_left() > 5.0:
        # Decide whether to start fresh or perturb best
        if attempts == 0 or random.random() < 0.2:
            # Start fresh
            current_edges = set()
            adj = [[False] * n for _ in range(n)]
            neighbors = [[] for _ in range(n)]
            neighbor_sets = [set() for _ in range(n)]
            
            random.seed(attempts)
            shuffled_edges = all_edges[:]
            random.shuffle(shuffled_edges)
        else:
            # Perturb best
            # Copy best graph
            current_edges = set(best_graph)
            adj = [[False] * n for _ in range(n)]
            neighbors = [[] for _ in range(n)]
            neighbor_sets = [set() for _ in range(n)]
            
            for u, v in best_graph:
                adj[u][v] = True
                adj[v][u] = True
                neighbors[u].append(v)
                neighbors[v].append(u)
                neighbor_sets[u].add(v)
                neighbor_sets[v].add(u)
                
            # Remove 2-5 random edges from current_edges
            remove_count = random.randint(2, 5)
            edges_list = list(current_edges)
            random.shuffle(edges_list)
            for i in range(min(remove_count, len(edges_list))):
                u, v = edges_list[i]
                current_edges.remove((u, v))
                adj[u][v] = False
                adj[v][u] = False
                # Remove from neighbors lists
                neighbors[u].remove(v)
                neighbors[v].remove(u)
                neighbor_sets[u].discard(v)
                neighbor_sets[v].discard(u)
            
            # Get remaining edges to try
            remaining_edges = [e for e in all_edges if e not in current_edges]
            random.shuffle(remaining_edges)
            shuffled_edges = remaining_edges

        graph_edges = list(current_edges)
        
        for u, v in shuffled_edges:
            if adj[u][v]:
                continue
            if not _check_c4(u, v, neighbors, neighbor_sets):
                # Add edge
                adj[u][v] = True
                adj[v][u] = True
                neighbors[u].append(v)
                neighbors[v].append(u)
                neighbor_sets[u].add(v)
                neighbor_sets[v].add(u)
                graph_edges.append((u, v))
                current_edges.add((u, v))
                
        if len(graph_edges) > best_edges:
            best_edges = len(graph_edges)
            best_graph = graph_edges[:]
            
        attempts += 1
        
    result = [[u, v, 0] for u, v in best_graph]
    return result
