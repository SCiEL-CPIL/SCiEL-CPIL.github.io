import time
import random

_PARAMS = ((32, 0.6, 601), (64, 0.55, 602))
_DEADLINE = 140.0

def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
    return adj, edges

def _solve_instance(n, p, seed, deadline):
    adj, edges = _build_graph(n, p, seed)
    # unassigned_adj[u] is a set of v such that (u,v) is unassigned
    unassigned_adj = [set(s) for s in adj]
    cliques = []
    
    # To speed up edge selection, we can pre-compute a list of all potential edges
    # and filter it as we go, or just iterate all pairs.
    # Iterating all pairs is O(n^2) = 4096 for n=64. Very fast.
    
    while True:
        if time.time() > deadline:
            break
            
        # Find the best edge (u, v) to start a clique
        # Heuristic: Maximize the number of common unassigned neighbors
        best_u, best_v = -1, -1
        best_common_size = -1
        
        # We only need to consider edges that are currently unassigned
        # An edge (u, v) is unassigned if v is in unassigned_adj[u]
        
        for u in range(n):
            if not unassigned_adj[u]:
                continue
            for v in unassigned_adj[u]:
                if u < v: # avoid duplicates
                    common = unassigned_adj[u] & unassigned_adj[v]
                    size = len(common)
                    if size > best_common_size:
                        best_common_size = size
                        best_u, best_v = u, v
                        
        if best_u == -1:
            # No unassigned edges left?
            # Check if any edge remains
            has_edges = False
            for u in range(n):
                if unassigned_adj[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            else:
                # Fallback: pick any edge
                # This shouldn't happen if the loop above is correct
                for u in range(n):
                    if unassigned_adj[u]:
                        v = next(iter(unassigned_adj[u]))
                        best_u, best_v = u, v
                        break

        # Start clique with best_u, best_v
        clique = [best_u, best_v]
        # Candidates are common neighbors
        candidates = unassigned_adj[best_u] & unassigned_adj[best_v]
        
        # Greedily expand
        while candidates:
            if time.time() > deadline:
                break
                
            # Find w in candidates that maximizes the new candidate set size
            # New candidates would be candidates & unassigned_adj[w]
            best_w = -1
            best_new_size = -1
            
            for w in candidates:
                new_cands = candidates & unassigned_adj[w]
                size = len(new_cands)
                # Tie-breaking: maybe prefer w with higher degree in candidates?
                # For now, just max size
                if size > best_new_size:
                    best_new_size = size
                    best_w = w
                    
            if best_w == -1:
                break
                
            # Add best_w to clique
            clique.append(best_w)
            # Update candidates
            candidates = candidates & unassigned_adj[best_w]
            
        # Remove all edges within the clique from unassigned_adj
        for i in range(len(clique)):
            u = clique[i]
            for j in range(i + 1, len(clique)):
                v = clique[j]
                # Remove v from u's adj and u from v's adj
                if v in unassigned_adj[u]:
                    unassigned_adj[u].discard(v)
                if u in unassigned_adj[v]:
                    unassigned_adj[v].discard(u)
                    
        cliques.append(sorted(clique))

    return cliques

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    results = []
    for (n, p, seed) in _PARAMS:
        if time.time() > deadline:
            # Fallback: single edge cliques
            adj, edges = _build_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        # Allocate remaining time minus a safety buffer for the next instance
        # There are 2 instances. Let's give each 50% of the time, minus 5s buffer.
        # Actually, just pass the global deadline. The function will stop if it exceeds.
        # But we need to ensure we have time for the second one.
        # Let's reserve 5 seconds for the second instance.
        sub_deadline = deadline - 5.0
        results.append(_solve_instance(n, p, seed, sub_deadline))
        
    return results
