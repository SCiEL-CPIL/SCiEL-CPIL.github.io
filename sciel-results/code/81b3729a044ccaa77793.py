"""Clique partition via greedy maximal-clique packing on remaining edges."""
import time
from itertools import combinations

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_graph(n, p, seed):
    import random
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

def _solve(n, p, seed):
    adj, edges = _make_graph(n, p, seed)
    
    # Initialize active edges
    active = set(edges)
    # Adjacency in the active subgraph
    # We need to update this as edges are removed.
    # For N=64, we can use bitmasks for speed, or sets.
    # Sets are easier to update.
    active_adj = [set() for _ in range(n)]
    for u, v in active:
        active_adj[u].add(v)
        active_adj[v].add(u)
        
    cliques = []
    
    # Precompute vertex order for deterministic tie-breaking
    # We want to find large cliques.
    
    while active and _remaining() > 1.0:
        # Pick a starting vertex with highest degree in active_adj
        # This helps find larger cliques
        if not active:
            break
            
        # Find vertex with max degree in active subgraph
        start_v = -1
        max_deg = -1
        for u in range(n):
            deg = len(active_adj[u])
            if deg > max_deg:
                max_deg = deg
                start_v = u
                
        if start_v == -1 or max_deg == 0:
            # No edges left? Or isolated vertices?
            # If max_deg == 0, there are no active edges incident to any vertex?
            # But active is not empty. This shouldn't happen if active_adj is consistent.
            # If active has edges, some vertex must have degree > 0.
            break

        # Build a maximal clique in the active subgraph starting from start_v
        clique = {start_v}
        # Candidates must be neighbors of all vertices in clique
        # Start with neighbors of start_v in active_adj
        cand = set(active_adj[start_v])
        
        while cand:
            # Pick candidate with highest degree within cand (or in active_adj?)
            # Standard greedy: pick v in cand with max |N(v) \cap cand|
            best_cand = None
            best_score = -1
            for v in cand:
                # Number of neighbors in cand
                score = len(active_adj[v] & cand)
                if score > best_score:
                    best_score = score
                    best_cand = v
            
            if best_cand is None:
                break
                
            clique.add(best_cand)
            # Update candidates: must be neighbors of best_cand AND in current cand
            cand = cand & active_adj[best_cand]
            
        # We have a maximal clique 'clique' in the active subgraph
        # All edges within 'clique' are in 'active'
        cl_list = sorted(clique)
        cliques.append(cl_list)
        
        # Remove all edges within this clique from active and active_adj
        for u, v in combinations(cl_list, 2):
            if (u, v) in active:
                active.remove((u, v))
                active_adj[u].discard(v)
                active_adj[v].discard(u)
            # Also check reverse if we stored both? 
            # We stored edges as (u,v) with u<v in 'edges', so in 'active'.
            # But active_adj is symmetric sets.
            # combinations gives u<v.
            # So (u,v) is the key.
            pass

    # Any remaining edges must be handled
    # They are single edges (cliques of size 2)
    # We can just add them.
    for u, v in active:
        cliques.append([u, v])
        
    return cliques

def construct():
    global _START
    _START = time.time()
    r1 = _solve(32, 0.6, 601)
    r2 = _solve(64, 0.55, 602)
    return [r1, r2]
