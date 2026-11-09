"""
Strategy: Greedy Set Partition over Maximal Cliques.

1. Generate all maximal cliques in the graph using Bron-Kerbosch algorithm.
2. Greedily select cliques that cover the most remaining edges.
3. Remove covered edges and repeat.

This avoids the pitfalls of building cliques greedily from a single edge, which might miss larger cliques that could be formed if we had picked a different starting edge or expansion path. By considering all maximal cliques, we ensure that if a large clique exists, it is a candidate for selection.
"""

import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch(adj, n, P, X, R):
    """
    Bron-Kerbosch algorithm to find all maximal cliques.
    adj: list of sets of neighbors.
    n: number of vertices.
    P: set of vertices that can be added to R.
    X: set of vertices that have been removed from P (and are not in R).
    R: current clique.
    Yields maximal cliques.
    """
    if not P and not X:
        yield R.copy()
        return
    
    # Pivot selection: choose vertex u in P ∪ X with maximum degree in P
    # This reduces the number of recursive calls
    u = None
    max_deg = -1
    for v in P | X:
        deg = len(P & adj[v])
        if deg > max_deg:
            max_deg = deg
            u = v
    
    # Recurse on P \ N(u)
    # Vertices in P that are NOT neighbors of u
    candidates = P - adj[u]
    
    for v in list(candidates):
        # New R = R ∪ {v}
        new_R = R | {v}
        # New P = P ∩ N(v)
        new_P = P & adj[v]
        # New X = X ∩ N(v)
        new_X = X & adj[v]
        
        yield from _bron_kerbosch(adj, n, new_P, new_X, new_R)
        
        # Move v from P to X
        P.discard(v)
        X.add(v)

def _find_maximal_cliques(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    P = set(range(n))
    X = set()
    R = set()
    
    cliques = []
    for clique in _bron_kerbosch(adj, n, P, X, R):
        if len(clique) >= 2:
            cliques.append(frozenset(clique))
            
    return cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    edge_set = set(edges)
    
    # Find all maximal cliques
    # For n=64, this might be slow if there are many, but let's try.
    # If it's too slow, we fall back to greedy peeling.
    start_time = time.time()
    
    try:
        maximal_cliques = _find_maximal_cliques(n, edges)
    except Exception:
        maximal_cliques = []
        
    if time.time() - start_time > 20:
        # Fallback to greedy peeling if clique enumeration is too slow
        return _greedy_fallback(n, edges)
        
    if not maximal_cliques:
        return [[u, v] for u, v in edges]
        
    # Greedy Set Partition
    remaining_edges = set(edge_set)
    partition = []
    
    # Pre-compute edge sets for each maximal clique
    clique_edges = []
    for clique in maximal_cliques:
        e_list = []
        for u, v in combinations(clique, 2):
            e = (u, v) if u < v else (v, u)
            if e in edge_set:
                e_list.append(e)
        if e_list:
            clique_edges.append((clique, set(e_list)))
            
    # Sort cliques by size descending to prioritize large ones?
    # Or just greedy pick max coverage each time.
    # Greedy pick max coverage is better.
    
    # To speed up, we can maintain a list of (clique, edges) and filter.
    # Since n is small, iterating over all cliques each step is okay if count is not huge.
    # If count is huge (e.g., > 10000), we might need to be careful.
    
    while remaining_edges:
        if not _deadline_ok():
            # Fallback: add remaining edges as individual cliques
            for e in remaining_edges:
                partition.append([e[0], e[1]])
            break
            
        best_clique = None
        best_edges = set()
        best_count = 0
        
        for clique, c_edges in clique_edges:
            # Count how many edges of this clique are still remaining
            # c_edges & remaining_edges
            common = c_edges & remaining_edges
            count = len(common)
            if count > best_count:
                best_count = count
                best_clique = clique
                best_edges = common
                
        if best_count == 0:
            # No maximal clique covers any remaining edge?
            # This implies the remaining edges form a graph with no triangles?
            # Or we missed some maximal cliques?
            # Bron-Kerbosch should find all.
            # If an edge is remaining, it must be in some maximal clique.
            # Unless the maximal clique was removed? No, we don't remove cliques, we remove edges.
            # So if an edge is remaining, it is in some maximal clique.
            # Why would best_count be 0?
            # Maybe the maximal clique containing this edge had other edges removed,
            # but the edge itself is still there.
            # The intersection `c_edges & remaining_edges` should contain this edge.
            # So best_count should be at least 1.
            # If this happens, it's a bug or edge case.
            # Fallback: pick any remaining edge.
            e = next(iter(remaining_edges))
            partition.append([e[0], e[1]])
            remaining_edges.remove(e)
            continue
            
        partition.append(list(best_clique))
        remaining_edges -= best_edges
        
    return partition

def _greedy_fallback(n, edges):
    # Simple greedy peeling as before, but optimized
    edge_set = set(edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    partition = []
    remaining = set(edge_set)
    
    while remaining:
        if not _deadline_ok():
            for e in remaining:
                partition.append([e[0], e[1]])
            break
            
        # Pick edge with max common neighbors
        best_e = None
        max_common = -1
        for u, v in remaining:
            common = len(adj[u] & adj[v])
            if common > max_common:
                max_common = common
                best_e = (u, v)
                
        if best_e is None:
            break
            
        u, v = best_e
        clique = {u, v}
        candidates = adj[u] & adj[v]
        
        # Greedy expansion
        while candidates:
            # Pick candidate with max degree in candidates
            best_w = None
            max_deg = -1
            for w in candidates:
                deg = len(adj[w] & candidates)
                if deg > max_deg:
                    max_deg = deg
                    best_w = w
                    
            if best_w is None:
                break
                
            clique.add(best_w)
            candidates = candidates & adj[best_w]
            
        partition.append(list(clique))
        
        # Remove edges
        for a, b in combinations(clique, 2):
            e = (a, b) if a < b else (b, a)
            remaining.discard(e)
            
    return partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
