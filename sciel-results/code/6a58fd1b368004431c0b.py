import time
import math
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur, strategy='min', time_limit=None):
        """
        Greedy clique partition.
        strategy: 'min' - pick edge with min common neighbors
                  'max' - pick edge with max common neighbors
                  'weighted' - pick edge with min (common_neighbors - weight * degree_balance)
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        # Precompute degrees for weighted strategy
        if strategy == 'weighted':
            degrees = [len(adj[i]) for i in range(n)]
        
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
            elif strategy == 'weighted':
                # Minimize common neighbors, but break ties by preferring edges 
                # where both endpoints have high remaining degree (more flexible)
                best_score = float('inf')
                for e in remaining_edges:
                    u, v = e
                    common = len(adj[u] & adj[v])
                    # Score: primarily common neighbors, secondarily - (deg[u] + deg[v])
                    # We want to minimize common, so high common is bad.
                    # Among same common, we might prefer to cover edges that are "harder" (low common) 
                    # or "easier" (high common)? 
                    # Actually, if common is low, the clique will be small. 
                    # If common is high, the clique will be large.
                    # Let's try a smooth surrogate: score = common - 0.1 * (deg[u] + deg[v])
                    score = common - 0.1 * (len(adj[u]) + len(adj[v]))
                    if score < best_score:
                        best_score = score
                        target_edge = e
            
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

    def local_improve(cliques, n, edges, seed_heur, time_limit):
        """
        Try to reduce the number of cliques by merging or moving edges.
        This is a heuristic local search.
        """
        # Build a map: edge -> clique index
        edge_to_clique = {}
        for i, C in enumerate(cliques):
            for u in range(len(C)):
                for v in range(u + 1, len(C)):
                    a, b = C[u], C[v]
                    e = (a, b) if a < b else (b, a)
                    edge_to_clique[e] = i
        
        # Check if we can merge two cliques into one
        # Two cliques C1 and C2 can be merged if C1 U C2 is a clique.
        # This is expensive to check all pairs.
        # Instead, try to move an edge from one clique to another if it fits.
        # Or simply restart with different seeds is often more effective for this problem size.
        
        # For now, let's just return the input. The main gain comes from diverse restarts.
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        strategies = ['min', 'max', 'weighted']
        
        while time.time() < time_limit:
            # Cycle through strategies
            strategy = strategies[attempt % len(strategies)]
            heur_seed = seed + attempt * 7919
            
            # Set a sub-time-limit for each greedy run to ensure we can try multiple
            # For n=32, greedy is fast. For n=64, it's slower.
            # Estimate time per run: ~0.1s for n=32, ~1s for n=64?
            # Let's allow up to 5s per run to be safe, but break earlier if possible.
            sub_limit = time.time() + 5.0
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

    # Allocate time
    # Instance 1: n=32, p=0.6. Edges ~ 32*31/2 * 0.6 = 297.
    # Instance 2: n=64, p=0.55. Edges ~ 64*63/2 * 0.55 = 1115.
    # Instance 2 is much harder. Give it more time.
    t1_limit = start_time + 20.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
