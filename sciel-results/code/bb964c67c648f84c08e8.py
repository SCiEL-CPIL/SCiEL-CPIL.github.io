import time
import sys
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    adj = [[False] * n for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = True
                adj[v][u] = True
                edges.append((u, v))
    return adj, edges

def _greedy_clique_partition(adj, n, edges, deadline):
    """
    Greedily partitions edges into cliques by repeatedly finding a large
    maximal clique in the residual graph.
    """
    remaining_edges = set(edges)
    # Pre-compute adjacency bitmasks for fast intersection?
    # For n=64, Python sets of ints are fine.
    
    # Helper to get neighbors in remaining graph
    # We can maintain a set of edges and check membership.
    
    partition = []
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: just add remaining edges as K2s
            for e in remaining_edges:
                partition.append(list(e))
            break
            
        # Find candidate seed edges
        # Heuristic: pick edges incident to vertices with high degree in residual graph
        # Calculate degrees in residual graph
        deg = {v: 0 for v in range(n)}
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
            
        # Get vertices sorted by degree descending
        sorted_vertices = sorted(range(n), key=lambda x: deg[x], reverse=True)
        
        best_clique = None
        best_size = 0
        
        # Try expanding from a few high-degree edges
        # Pick the top K vertices, and try edges between them or from them to others
        candidates = []
        for u in sorted_vertices[:10]:
            if deg[u] == 0: continue
            # Find a neighbor v in remaining graph
            # We need to iterate remaining_edges to find neighbors of u
            # This is O(E) per u, which is okay for small n.
            neighbors_of_u = []
            for (a, b) in remaining_edges:
                if a == u:
                    neighbors_of_u.append(b)
                elif b == u:
                    neighbors_of_u.append(a)
            
            # Try a few neighbors
            for v in neighbors_of_u[:5]:
                candidates.append((u, v))
        
        # If no candidates found (should not happen if remaining_edges not empty), fallback to any edge
        if not candidates and remaining_edges:
            candidates = [next(iter(remaining_edges))]
            
        # For each candidate edge, try to expand to a maximal clique
        for u, v in candidates:
            if time.time() > deadline:
                break
                
            # Greedy expansion
            clique = [u, v]
            # Common neighbors of u and v in remaining graph
            # A vertex w is a common neighbor if (u,w) and (v,w) are in remaining_edges
            common = []
            for w in range(n):
                if w == u or w == v: continue
                if (min(u,w), max(u,w)) in remaining_edges and (min(v,w), max(v,w)) in remaining_edges:
                    common.append(w)
            
            # Greedily add vertices from common
            # We want to add w if it is connected to all current clique members
            # Since we start with u,v, and w is common neighbor, w is connected to u and v.
            # As we add w1, we must ensure new candidates w2 are connected to w1.
            
            while common:
                # Pick vertex in common with highest degree in the "potential" set
                # Or just pick the one with highest degree in remaining graph
                # Let's pick the one with highest deg in remaining graph
                w_best = max(common, key=lambda x: deg[x])
                
                # Check if w_best is connected to all in clique
                valid = True
                for c in clique:
                    e = (min(w_best, c), max(w_best, c))
                    if e not in remaining_edges:
                        valid = False
                        break
                
                if valid:
                    clique.append(w_best)
                    common.remove(w_best)
                    # Update common: remove any w not connected to w_best
                    new_common = []
                    for w in common:
                        e = (min(w_best, w), max(w_best, w))
                        if e in remaining_edges:
                            new_common.append(w)
                    common = new_common
                else:
                    # If not valid, we can't add it. 
                    # In a greedy maximal clique, if we can't add the best one, 
                    # we might skip it or try the next. 
                    # For simplicity and to ensure maximality, we should probably 
                    # just remove it from consideration if it blocks others? 
                    # Actually, if it's not connected to clique, it can't be in the clique.
                    # So we just remove it.
                    common.remove(w_best)
                    # But wait, if we remove it, we don't filter others by it.
                    # This is a standard greedy maximal clique algorithm.
                    pass

            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
                
        if best_clique is None or best_size < 2:
            # Should not happen if remaining_edges is not empty
            # Fallback to single edge
            u, v = next(iter(remaining_edges))
            best_clique = [u, v]
            
        # Add best_clique to partition
        partition.append(best_clique)
        
        # Remove edges in best_clique from remaining_edges
        for i in range(len(best_clique)):
            for j in range(i + 1, len(best_clique)):
                u, v = best_clique[i], best_clique[j]
                e = (min(u, v), max(u, v))
                if e in remaining_edges:
                    remaining_edges.remove(e)
                    
    return partition

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback to baseline if time runs out
            rng = Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            results.append([list(e) for e in edges])
            continue
            
        adj, edges = _random_graph(n, p, seed)
        
        partition = _greedy_clique_partition(adj, n, edges, deadline)
        results.append(partition)
        
    return results
