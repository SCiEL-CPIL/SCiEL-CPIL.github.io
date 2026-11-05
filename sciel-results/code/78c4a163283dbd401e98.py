import time
import sys

# EVOLVE-BLOCK-START
def construct():
    # We aim for n=32, which is the best known bound mentioned.
    # Construction: 8 disjoint copies of K4.
    # K4 has 4 vertices and 6 edges.
    # Total vertices: 8 * 4 = 32.
    # Total edges: 8 * 6 = 48.
    # Independence number: 8 * 1 = 8. (Condition: < 9, so 8 is valid)
    # C5 check: K4 is C5-free. Disjoint union of C5-free graphs is C5-free.
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
