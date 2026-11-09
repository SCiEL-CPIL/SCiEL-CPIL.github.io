"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        edges = []
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def max_clique_from_vertex(v, candidates, adj):
        """Find maximum clique using Bron-Kerbosch with pivot, starting from vertex v."""
        best = [v]
        
        def bronk(r, p, x):
            nonlocal best
            if not p and not x:
                if len(r) > len(best):
                    best = list(r)
                return
            if time.time() > deadline - 5:
                return
            
            # Pivot selection: choose u from P union X with max degree
            all_candidates = p | x
            if all_candidates:
                # Choose pivot with max degree in p
                pivot = max(all_candidates, key=lambda u: len(p & adj[u]))
                candidates_to_try = p - adj[pivot]
            else:
                candidates_to_try = p
            
            for w in list(candidates_to_try):
                bronk(r | {w}, p & adj[w], x & adj[w])
                p -= {w}
                x |= {w}
        
        bronk({v}, candidates, set())
        return best

    def solve_instance(n, p, seed, time_budget):
        edges, adj = instance(n, p, seed)
        if not edges:
            return []
        
        edge_set = set(edges)
        cliques = []
        
        # Strategy: repeatedly find a large clique from remaining edges
        # Use a greedy approach: pick an edge, find max clique containing it, remove those edges
        remaining_edges = set(edges)
        remaining_adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in remaining_edges:
                    remaining_adj[u].add(v)
                    remaining_adj[v].add(u)
        
        while remaining_edges and time.time() < time_budget:
            # Pick an edge to start from - prefer edges with high common neighbor count
            # to find larger cliques
            best_edge = None
            best_common = -1
            
            # Sample some edges to find a good starting point
            edge_list = list(remaining_edges)
            if len(edge_list) <= 20:
                sample = edge_list
            else:
                sample = random.sample(edge_list, 20)
            
            for e in sample:
                u, v = e
                common = len(remaining_adj[u] & remaining_adj[v])
                if common > best_common:
                    best_common = common
                    best_edge = e
            
            if best_edge is None:
                # Fallback: just take one edge
                u, v = edge_list[0]
                cliques.append([u, v])
                remaining_edges.discard((u, v))
                remaining_adj[u].discard(v)
                remaining_adj[v].discard(u)
                continue
            
            u0, v0 = best_edge
            # Find max clique containing both u0 and v0
            # Start with {u0, v0}, find common neighbors, run Bron-Kerbosch
            initial_clique = {u0, v0}
            candidates = remaining_adj[u0] & remaining_adj[v0]
            candidates -= {u0, v0}
            
            # Run Bron-Kerbosch to find max clique in candidates that can be added
            best_clique = {u0, v0}
            
            def bronk_expand(r, p, x):
                nonlocal best_clique
                if not p and not x:
                    if len(r) > len(best_clique):
                        best_clique = set(r)
                    return
                if time.time() > time_budget - 2:
                    return
                
                # Pivot
                all_c = p | x
                if all_c:
                    pivot = max(all_c, key=lambda u: len(p & remaining_adj[u]))
                    to_try = p - remaining_adj[pivot]
                else:
                    to_try = p
                
                for w in list(to_try):
                    bronk_expand(r | {w}, p & remaining_adj[w], x & remaining_adj[w])
                    p -= {w}
                    x |= {w}
            
            bronk_expand({u0, v0}, candidates, set())
            
            # Remove all edges within best_clique
            clique_list = list(best_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        remaining_adj[a].discard(b)
                        remaining_adj[b].discard(a)
            
            cliques.append(clique_list)
        
        # Add any remaining edges as single-edge cliques
        for e in remaining_edges:
            cliques.append(list(e))
        
        return cliques

    # Solve both instances
    # Allocate time: 60% to first instance (smaller, n=32), 40% to second (n=64)
    total_time = deadline - start_time
    time_budget1 = start_time + total_time * 0.45
    time_budget2 = deadline - 2
    
    result1 = solve_instance(32, 0.6, 601, time_budget1)
    result2 = solve_instance(64, 0.55, 602, time_budget2)
    
    return [result1, result2]

# EVOLVE-BLOCK-END
