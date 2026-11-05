import time
import random
import sys

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Returns the known valid n=32 construction (8 disjoint K4s).
    The search for n=33 is removed to ensure speed and reliability,
    as n=32 is the best known lower bound and n=33 is likely impossible
    or extremely difficult to find within the time limit.
    """
    # Fallback: 8 disjoint K4s, n=32
    n_fallback = 32
    edges_fallback = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges_fallback.append([verts[i], verts[j]])
    
    return {'n': n_fallback, 'edges': edges_fallback}

if __name__ == "__main__":
    r = construct()
    print(r['n'])
