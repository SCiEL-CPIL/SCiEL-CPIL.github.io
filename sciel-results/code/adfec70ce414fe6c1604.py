"""
Constructs a C5-free graph with no independent set of size 9.
Uses the known optimal structure of 8 disjoint copies of K4, achieving n=32.
"""
import time

def construct():
    # The best known lower bound for R(C5, K9) > n is n=32.
    # This is achieved by the disjoint union of 8 copies of K4.
    # n = 32
    # alpha = 8 (one vertex from each K4)
    # C5-free (K4 is C5-free, disjoint union is C5-free)
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
