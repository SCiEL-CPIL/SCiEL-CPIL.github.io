"""K2,3-free graph on 20 vertices: maximize edges via greedy + local search."""
import time
import random

N = 20
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    best_edges = None
    best_count = 0
    
    def make_greedy():
        adj = [set() for _ in range(N)]
        adj_set = set()
        
        def can_add(u, v):
            if u == v or (u, v) in adj_set or (v, u) in adj_set:
                return False
            for x in adj[v]:
                if x == u:
                    continue
                if len(adj[u] & adj[x]) >= 2:
                    return False
            for x in adj[u]:
                if x == v:
                    continue
                if len(adj[v] & adj[x]) >= 2:
                    return False
            return True
        
        def add(u, v):
            adj[u].add(v)
            adj[v].add(u)
            if u < v:
                adj_set.add((u, v))
            else:
                adj_set.add((v, u))
        
        def remove(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            if u < v:
                adj_set.discard((u, v))
            else:
                adj_set.discard((v, u))
        
        def local_search():
            improved = True
            while improved and (time.time() - start_time) < DEADLINE:
                improved = False
                for (a, b) in all_edges:
                    if (a, b) not in adj_set and can_add(a, b):
                        add(a, b)
                        improved = True
                        if (time.time() - start_time) > DEADLINE:
                            break
                if (time.time() - start_time) > DEADLINE:
                    break
            return adj, len(adj_set)
        
        # Greedy build
        edges_copy = all_edges[:]
        rng.shuffle(edges_copy)
        for (u, v) in edges_copy:
            if can_add(u, v):
                add(u, v)
        
        # Local search to fill in
        adj, count = local_search()
        return adj, count, adj_set
    
    num_restarts = 100
    for i in range(num_restarts):
        if (time.time() - start_time) > DEADLINE:
            break
        adj, count, adj_set = make_greedy()
        if count > best_count:
            best_count = count
            best_edges = sorted(list(adj_set))
    
    # Additional improvement attempts
    if best_edges is not None and (time.time() - start_time) < DEADLINE - 15:
        adj = [set() for _ in range(N)]
        adj_set = set(best_edges)
        for (u, v) in adj_set:
            adj[u].add(v)
            adj[v].add(u)
        
        def can_add2(u, v):
            if u == v or (u, v) in adj_set or (v, u) in adj_set:
                return False
            for x in adj[v]:
                if x == u:
                    continue
                if len(adj[u] & adj[x]) >= 2:
                    return False
            for x in adj[u]:
                if x == v:
                    continue
                if len(adj[v] & adj[x]) >= 2:
                    return False
            return True
        
        def add2(u, v):
            adj[u].add(v)
            adj[v].add(u)
            if u < v:
                adj_set.add((u, v))
            else:
                adj_set.add((v, u))
        
        def remove2(u, v):
            adj[u].discard(v)
            adj[v].discard(u)
            if u < v:
                adj_set.discard((u, v))
            else:
                adj_set.discard((v, u))
        
        # Try random perturbations
        for trial in range(500):
            if (time.time() - start_time) > DEADLINE:
                break
            edge_list = list(adj_set)
            if not edge_list:
                break
            (a, b) = rng.choice(edge_list)
            remove2(a, b)
            # Try to add as many edges as possible
            improved = True
            while improved and (time.time() - start_time) < DEADLINE:
                improved = False
                for (x, y) in all_edges:
                    if (x, y) not in adj_set and can_add2(x, y):
                        add2(x, y)
                        improved = True
                        if (time.time() - start_time) > DEADLINE:
                            break
                if (time.time() - start_time) > DEADLINE:
                    break
            if len(adj_set) > best_count:
                best_count = len(adj_set)
                best_edges = sorted(list(adj_set))
    
    if best_edges is None:
        best_edges = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    
    return best_edges
