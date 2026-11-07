"""Rewritten program: pack edge-disjoint Hamiltonian cycles on a shared vertex set."""
import time
import random
import sys
import itertools
sys.setrecursionlimit(1000000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
    n = P['n']
    # The host graph is the circulant graph C_21(1, 2, 5).
    # It is 6-regular, so it can be decomposed into 3 edge-disjoint Hamiltonian cycles.
    # We construct them using step vectors 1, 2, and 5 modulo 21.
    
    cycle1 = [(i * 1) % n for i in range(n)]
    cycle2 = [(i * 2) % n for i in range(n)]
    cycle3 = [(i * 5) % n for i in range(n)]
    
    # Verify validity internally to be safe, though mathematically guaranteed
    edge_set = set((min(a, b), max(a, b)) for a, b in P['edges'])
    used_edges = set()
    valid = True
    for cycle in [cycle1, cycle2, cycle3]:
        if len(set(cycle)) != n:
            valid = False
            break
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            e = (min(u, v), max(u, v))
            if e not in edge_set or e in used_edges:
                valid = False
                break
            used_edges.add(e)
        if not valid:
            break
            
    if valid:
        return [cycle1, cycle2, cycle3]
    else:
        # Fallback to a single cycle if something unexpected happens (should not occur)
        return [list(range(n))]
