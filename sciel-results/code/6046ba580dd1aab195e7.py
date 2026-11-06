import time
import random

def construct():
    # Known valid construction for n=32: 8 disjoint copies of K4.
    # Each K4 has 4 vertices and 6 edges.
    # Total vertices: 8 * 4 = 32.
    # Independence number of K4 is 1.
    # Independence number of disjoint union is sum of independence numbers.
    # alpha = 8 * 1 = 8 < 9.
    # C5-free: K4 has no cycles of length > 3. Disjoint union of C5-free graphs is C5-free.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
