"""
Constructs a graph on vertices 0..11 with minimum degree 3, girth at least 4,
and exactly 2 distinct simple cycle lengths (4 and 6).

This version uses a bipartite graph construction model.
We define two partitions U and V of size 6.
We construct two disjoint K_{3,3} components.
This ensures min degree 3, triangle-free (bipartite), and cycle lengths {4, 6}.
"""

import time

_START_TIME = None

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Safety deadline
    DEADLINE = _START_TIME + 85

    # Bipartite sets
    U = [0, 1, 2, 3, 4, 5]
    V = [6, 7, 8, 9, 10, 11]
    
    # We will form two K_{3,3} components.
    # Component 1: U[0:3] connected to V[0:3]
    # Component 2: U[3:6] connected to V[3:6]
    
    edges = []
    
    # Component 1
    for u in U[:3]:
        for v in V[:3]:
            edges.append([u, v])
            
    # Component 2
    for u in U[3:6]:
        for v in V[3:6]:
            edges.append([u, v])
            
    # Validation
    final_edges = []
    seen_pairs = set()
    
    for e in edges:
        u, v = e
        u_int = int(u)
        v_int = int(v)
        
        if u_int == v_int:
            continue
            
        pair_key = (min(u_int, v_int), max(u_int, v_int))
        
        if pair_key in seen_pairs:
            continue
            
        seen_pairs.add(pair_key)
        final_edges.append([u_int, v_int])
        
    final_edges.sort()
    
    if time.time() > DEADLINE:
        return final_edges
        
    return final_edges
