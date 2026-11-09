import time
import itertools
import random

def construct():
    """
    Constructs a graph with 8 disjoint copies of K4.
    n = 32.
    No 5-cycle: Each component is K4 (max cycle length 4). Disjoint components do not form cycles.
    No independent set of 9: Alpha(K4) = 1. Total alpha = 8 * 1 = 8 < 9.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well within 120s limit
    
    n = 32
    edges = []
    for b in range(8):
        if time.time() > deadline:
            break
        base = 4 * b
        for i in range(4):
            if time.time() > deadline:
                break
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
