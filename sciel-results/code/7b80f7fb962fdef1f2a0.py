# EVOLVE-BLOCK-START
import time
from random import Random
from itertools import combinations

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _find_maximal_cliques(adj):
    """
    Find all maximal cliques in the graph using Bron-Kerbosch algorithm with pivot.
    """
    n = len(adj)
    cliques = []
    
    # Use sets for P, X, R
    # R: current clique
    # P: candidates that can be added
    # X: candidates that have already been processed (to ensure maximality)
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P union X to minimize branching
        # Heuristic: pick vertex with max degree in P
        u = None
        max_deg = -1
        for v in P:
            deg = len(adj[v] & P)
            if deg > max_deg:
                max_deg = deg
                u = v
        if u is None:
            for v in X:
                deg = len(adj[v] & P)
                if deg > max_deg:
                    max_deg = deg
                    u = v
        
        if u is not None:
            # Recurse on P \ N(u)
            candidates = P - adj[u]
        else:
            candidates = P
            
        for v in list(candidates):
            # Add v to R
            new_R = R | {v}
            # New P = P \ N(v)
            new_P = P - adj[v]
            # New X = X \ N(v)
            new_X = X - adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Move v from P to X
            P.discard(v)
            X.add(v)

    # Initial call
    bron_kerbosch(frozenset(), set(range(n)), set())
    return cliques

def _greedy_cover(cliques, adj, deadline):
    """
    Greedily select cliques to cover all edges.
    """
    n = len(adj)
    # Build list of all edges
    all_edges = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                all_edges.add((u, v))
                
    if not all_edges:
        return []
        
    # Map each clique to its edges
    clique_edges = []
    for c in cliques:
        c_list = list(c)
        edges = set()
        for i in range(len(c_list)):
            for j in range(i+1, len(c_list)):
                u, v = c_list[i], c_list[j]
                if (min(u,v), max(u,v)) in all_edges:
                    edges.add((min(u,v), max(u,v)))
        if edges:
            clique_edges.append((c, edges))
            
    # Sort by number of edges descending initially
    clique_edges.sort(key=lambda x: len(x[1]), reverse=True)
    
    remaining_edges = set(all_edges)
    selected = []
    
    # For efficiency, we maintain a list of (clique, current_edge_count)
    # But updating counts is expensive. 
    # Given n=64, number of cliques is likely < 10000.
    # Number of edges is < 2000.
    # Naive check: O(N_cliques * N_edges) per step.
    # 10000 * 2000 = 20M ops per step. 
    # If we have 2000 steps, that's 40B ops. Too slow.
    
    # Optimization: Only consider cliques that still have remaining edges.
    # We can filter the list in each iteration.
    
    while remaining_edges and time.time() < deadline:
        best_clique = None
        best_count = 0
        best_edges = None
        
        # Iterate through all candidate cliques
        for c, edges in clique_edges:
            # Count how many of its edges are in remaining_edges
            # Intersection size
            count = len(edges & remaining_edges)
            if count > best_count:
                best_count = count
                best_clique = c
                best_edges = edges
        
        if best_count == 0:
            # Should not happen if we have all maximal cliques, 
            # but if we missed some, we break
            break
            
        selected.append(list(best_clique))
        remaining_edges -= best_edges
        
    # Cover any remaining edges with K2
    if remaining_edges:
        for edge in remaining_edges:
            selected.append([edge[0], edge[1]])
            
    return selected

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
            
        adj = _build_graph(n, p, seed)
        
        # Find maximal cliques
        # Bron-Kerbosch can be slow on dense graphs if not careful, 
        # but n=64 is small.
        try:
            cliques = _find_maximal_cliques(adj)
        except Exception:
            cliques = []
            
        # If finding cliques takes too long or fails, fallback to greedy K3 extension
        if time.time() > deadline - 10:
             # Fallback to simpler method if stuck in clique finding
             # But _find_maximal_cliques is recursive, hard to interrupt easily.
             # We rely on n=64 being fast enough.
             pass
             
        partition = _greedy_cover(cliques, adj, deadline)
        results.append(partition)
        
    return results
# EVOLVE-BLOCK-END
