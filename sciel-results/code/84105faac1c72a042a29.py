"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    n = P['n']
    # Partition vertices into two sets of size n/2
    # Set A: 0 to n//2 - 1
    # Set B: n//2 to n - 1
    half = n // 2
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
