import time
import math
from random import Random

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we return well before the 120s limit
    deadline = start_time + 90.0

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur, strategy='max', time_limit=None):
        """
        Greedy clique partition.
        strategy: 'min' - pick edge with min common neighbors
                  'max' - pick edge with max common neighbors
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            if time_limit is not None and time.time() > time_limit:
                break
                
            target_edge = None
            
            if strategy == 'min':
                min_common = float('inf')
                for e in remaining_edges:
                    u, v = e
                    common = len(adj[u] & adj[v])
                    if common < min_common:
                        min_common = common
                        target_edge = e
                        if common == 0:
                            break
            elif strategy == 'max':
                max_common = -1
                for e in remaining_edges:
                    u, v = e
                    common = len(adj[u] & adj[v])
                    if common > max_common:
                        max_common = common
                        target_edge = e
            else:
                target_edge = next(iter(remaining_edges))
            
            if target_edge is None:
                target_edge = next(iter(remaining_edges))
            
            u, v = target_edge
            # Find maximal clique containing u, v
            C = [u, v]
            P = adj[u] & adj[v]
            
            while P:
                # Pick vertex in P with max degree in P (greedy coloring heuristic)
                # To speed up, we can sample or just iterate. For n<=64, iterating is fine.
                max_d = -1
                best_cands = []
                for x in P:
                    d = len(adj[x] & P)
                    if d > max_d:
                        max_d = d
                        best_cands = [x]
                    elif d == max_d:
                        best_cands.append(x)
                
                w = rng_heur.choice(best_cands)
                
                C.append(w)
                P = P & adj[w]
            
            # Remove edges of C from remaining_edges and adj
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
            
            cliques.append(C)
            
        # If we broke early due to time, we have remaining edges. 
        # We must cover them. The simplest valid cover is single edges.
        # This ensures validity but might increase clique count significantly.
        # However, the contract requires a valid partition.
        if remaining_edges:
            for e in remaining_edges:
                cliques.append(list(e))
        
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        # Use 'max' and 'min' strategies. 'max' tends to form larger cliques.
        strategies = ['max', 'min']
        
        while time.time() < time_limit:
            strategy = strategies[attempt % len(strategies)]
            heur_seed = seed + attempt * 7919
            
            # Set a sub-time-limit for each greedy run
            # Short runs (2s) allow for many diverse restarts
            sub_limit = time.time() + 2.0
            if sub_limit > time_limit:
                sub_limit = time_limit
            
            cliques = greedy_partition(n, edges, heur_seed, strategy, sub_limit)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: single edge cliques
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time conservatively
    # Instance 1: n=32. Fast. Give it 10s.
    t1_limit = start_time + 10.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    # Instance 2: n=64. Slower. Give it the rest, up to 80s.
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
