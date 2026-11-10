"""Clique partition of split graphs with greedy max-clique extraction."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency: clique vertices are 0..k-1, independent set is k..n-1
        # For clique vertices, they're all connected to each other.
        # For independent set vertices, they connect only to clique vertices (with probability p).
        
        # Represent edges as sets
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Build adjacency for all vertices
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # For clique vertices (0..k-1), they form a complete subgraph
        # For independent set vertices (k..n-1), no edges between them
        
        # Greedy clique partition: repeatedly find a large clique, add it, remove its edges
        cliques = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if time.time() > deadline - 5:
                # Fallback: just list each edge as a clique
                for (u, v) in remaining_edges:
                    cliques.append([u, v])
                break
            
            # Find a large clique among remaining edges
            # Strategy: pick a vertex with high degree in remaining graph, try to extend
            # For split graphs, max cliques have special structure
            
            # Find max degree vertex in remaining graph
            rem_deg = [0] * n
            rem_adj = [set() for _ in range(n)]
            for (u, v) in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Pick vertex with max degree
            max_v = max(range(n), key=lambda x: rem_deg[x])
            
            # Try to build a large clique containing max_v
            # Candidates are neighbors of max_v in remaining graph
            candidates = sorted(rem_adj[max_v], key=lambda x: -rem_deg[x])
            
            # Greedily build clique
            current_clique = [max_v]
            available = set(candidates)
            
            for cand in candidates:
                if cand not in available:
                    continue
                # Check if cand is connected to all in current_clique
                can_add = True
                for vc in current_clique:
                    if cand not in rem_adj[vc]:
                        can_add = False
                        break
                if can_add:
                    current_clique.append(cand)
                    # Remove from available those not connected to cand
                    available = {x for x in available if x in rem_adj[cand]}
            
            # Also try without max_v - maybe a clique not containing max_v is larger
            # But for now, use this greedy approach
            
            if len(current_clique) < 2:
                # Shouldn't happen if remaining_edges is non-empty
                # Fallback: take any edge
                u, v = next(iter(remaining_edges))
                current_clique = [u, v]
            
            cliques.append(current_clique)
            
            # Remove all edges covered by this clique
            clique_set = set(current_clique)
            new_remaining = set()
            for (u, v) in remaining_edges:
                if u in clique_set and v in clique_set:
                    continue
                new_remaining.add((u, v))
            remaining_edges = new_remaining
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
