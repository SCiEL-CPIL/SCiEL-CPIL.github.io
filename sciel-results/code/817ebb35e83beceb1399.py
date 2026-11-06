import time
import random
import numpy as np

def _has_c5(adj: list[set]) -> bool:
    n = len(adj)
    # Check for C5 subgraph using ordered vertices u < v < w < x < y
    # Edges: (u,v), (v,w), (w,x), (x,y), (y,u)
    for u in range(n):
        for v in adj[u]:
            if v <= u: continue
            for w in adj[v]:
                if w <= v or w == u: continue
                for x in adj[w]:
                    if x <= w or x in (u, v): continue
                    for y in adj[x]:
                        if y <= x or y in (u, v, w): continue
                        if y in adj[u]:
                            return True
    return False

def _has_independent_set_size_k(adj: list[set], k: int) -> bool:
    n = len(adj)
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    
    found = [False]
    
    def search(sz, candidates):
        if found[0]: return
        if sz >= k:
            found[0] = True
            return
        if not candidates: return
        if sz + len(candidates) < k: return
        
        # Heuristic: pick vertex with max degree in comp induced by candidates
        # Sort candidates by degree in comp
        v = max(candidates, key=lambda x: len(comp[x] & candidates))
        
        # Include v
        nc = candidates & comp[v]
        search(sz + 1, nc)
        if found[0]: return
        
        # Exclude v
        c2 = candidates - {v}
        search(sz, c2)
        
    search(0, frozenset(range(n)))
    return found[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Default: 8 disjoint K4s
    def make_default():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return n, edges

    best_n, best_edges = make_default()
    
    # Verification of default
    adj = [set() for _ in range(best_n)]
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
    if _has_c5(adj) or _has_independent_set_size_k(adj, 9):
        # Should not happen, but fallback to empty?
        return {'n': 1, 'edges': []}

    random.seed(42)
    
    # Search for a 33-vertex graph
    # Strategy: Random 32-vertex base, then add 1 vertex.
    # Or Random 33-vertex graph directly and prune?
    # Pruning is harder. Adding is easier to verify.
    
    for iter_count in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Generate random 32-vertex graph
        n_base = 32
        adj_base = [set() for _ in range(n_base)]
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if random.random() < 0.5:
                    adj_base[i].add(j)
                    adj_base[j].add(i)
        
        # Check C5-free
        if _has_c5(adj_base):
            continue
            
        # Check alpha < 9
        if _has_independent_set_size_k(adj_base, 9):
            continue
            
        # Base is valid. Try to add vertex 32.
        # Try several random neighbor sets for the new vertex.
        for trial in range(20):
            # New vertex 32
            # Neighbors S subset of 0..31
            S = set()
            for i in range(n_base):
                if random.random() < 0.5:
                    S.add(i)
            
            # Construct new graph
            adj_new = [set(s) for s in adj_base] + [set(S)]
            for u in S:
                adj_new[u].add(32)
            
            # Check C5
            if _has_c5(adj_new):
                continue
                
            # Check alpha
            if _has_independent_set_size_k(adj_new, 9):
                continue
                
            # Valid!
            edges_new = []
            for i in range(n_base):
                for j in adj_base[i]:
                    if i < j:
                        edges_new.append([i, j])
            for u in S:
                edges_new.append([32, u])
                
            return {'n': 33, 'edges': edges_new}

    return {'n': best_n, 'edges': best_edges}
