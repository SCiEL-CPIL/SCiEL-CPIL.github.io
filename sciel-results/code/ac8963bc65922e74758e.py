"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    # The graph is fixed. We have pre-computed 3 edge-disjoint Hamiltonian cycles.
    # C1: Perimeter
    # C2: Jump pattern
    # C3: Remaining edges form a single cycle
    
    c1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    c2 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1, 3, 19, 17, 15, 13, 11, 9, 7, 5]
    c3 = [0, 16, 11, 6, 1, 17, 12, 7, 2, 18, 13, 8, 3, 5, 10, 15, 20, 4, 9, 14, 19]
    
    # Verify validity just in case (optional but safe)
    adj = _build_adj(P['n'], P['edges'])
    used_edges = set()
    
    for c in [c1, c2, c3]:
        if len(c) != P['n'] or len(set(c)) != P['n']:
            continue
        for i in range(len(c)):
            u = c[i]
            v = c[(i + 1) % len(c)]
            e = (min(u, v), max(u, v))
            if e not in adj[u]: # adj[u] is a set of neighbors, e is a tuple. Check if v in adj[u]
                pass
            if v not in adj[u]:
                return [c1] # Fallback
            if e in used_edges:
                return [c1] # Fallback
            used_edges.add(e)
            
    return [list(c1), list(c2), list(c3)]

# EVOLVE-BLOCK-END
