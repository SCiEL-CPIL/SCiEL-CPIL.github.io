import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 108

def _can_add_edge_fast(u, v, adj, n=40):
    """
    Check if adding edge (u,v) creates a C6 in graph represented by adj (list of sets).
    A C6 is u-a-b-c-d-v where a,b,c,d are distinct and not u or v.
    We check all paths of length 5 from u to v in the current graph (without edge u-v).
    If such a path exists, adding (u,v) creates a C6.
    """
    # We need a simple path u -> a -> b -> c -> d -> v of length 5
    # a in N(u), b in N(a)\{u}, c in N(b)\{u,a}, d in N(c)\{u,a,b}, v in N(d)\{u,a,b,c}
    
    # Optimization: iterate over a in N(u)
    # For each a, iterate over b in N(a) \ {u}
    # For each b, iterate over c in N(b) \ {u, a}
    # For each c, check if any d in N(c) \ {u, a, b} has v in N(d) \ {u, a, b, c}
    
    # To speed up, we can precompute for each c, the set of vertices that are at distance 1 from c
    # and have v as a neighbor, excluding u, a, b.
    # Actually, let's just do the nested loops but with sets for O(1) lookup.
    
    for a in adj[u]:
        if a == v:
            continue
        for b in adj[a]:
            if b == u or b == a or b == v:
                continue
            for c in adj[b]:
                if c == u or c == a or c == b or c == v:
                    continue
                # Now check if there exists d in N(c) such that d != u,a,b,c,v and v in N(d)
                # d in adj[c] and v in adj[d] and d not in {u,a,b,c}
                # Note: d can be v? No, d is internal.
                # d must be neighbor of c and neighbor of v.
                # So d in adj[c] & adj[v], excluding {u, a, b, c}
                # Wait, d must also not be v.
                candidates = adj[c] & adj[v]
                # Remove u, a, b, c, v
                candidates.discard(u)
                candidates.discard(a)
                candidates.discard(b)
                candidates.discard(c)
                candidates.discard(v)
                if candidates:
                    return False
    return True

def _greedy_with_local_search(n=40, seed=0, deadline=None):
    if deadline is None:
        deadline = _DEADLINE
    rng = random.Random(seed)
    
    adj = [set() for _ in range(n)]
    edges = set()
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_edges)
    
    # Greedy Phase: add edges if valid
    for u, v in all_edges:
        if time.time() > deadline - 2:
            break
        if _can_add_edge_fast(u, v, adj, n):
            adj[u].add(v)
            adj[v].add(u)
            edges.add((u, v))
            
    # Local Search Phase: Iterated Local Search
    # Try to add non-edges, and if stuck, remove some edges and retry
    non_edges = [(u, v) for u, v in all_edges if (u, v) not in edges]
    rng.shuffle(non_edges)
    
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for u, v in non_edges:
            if (u, v) in edges:
                continue
            if _can_add_edge_fast(u, v, adj, n):
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
                improved = True
                if time.time() > deadline - 3:
                    break
    
    return edges, adj

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 108
    
    best_edges = set()
    best_count = 0
    n = 40
    
    # Use multiple seeds with ILS
    for seed in range(10000):
        if time.time() > _DEADLINE - 2:
            break
        
        try:
            edges, adj = _greedy_with_local_search(n, seed)
            count = len(edges)
            if count > best_count:
                best_count = count
                best_edges = edges
                if count >= 105:
                    break
        except Exception:
            continue
            
    return [[u, v] for u, v in best_edges]
