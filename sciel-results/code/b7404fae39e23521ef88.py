import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 3 and g == 5:
        return _petersen_graph()

    if k == 5 and g == 9:
        # Try Cayley graph on Z_n for n = 1116
        n_target = 1116
        # Generators: +/- a, +/- b, 558
        # a, b in Z_n, a != 0, b != 0, a != b, a != -b, etc.
        
        # We will try random a, b
        for _ in range(1000): # Try 1000 random pairs
            if time.time() > deadline - 5:
                break
            
            a = random.randint(1, n_target - 1)
            b = random.randint(1, n_target - 1)
            
            # Ensure distinct and non-zero
            if a == b or a == n_target - b or a == 0 or b == 0:
                continue
            
            # Also ensure a != 558, b != 558
            if a == n_target // 2 or b == n_target // 2:
                continue
                
            # Generators: a, -a, b, -b, 558
            # -a = n_target - a
            # -b = n_target - b
            
            # Build edges
            edges = []
            # For each vertex i, neighbors are i+a, i-a, i+b, i-b, i+558
            # But we only add each edge once.
            # i+a edge is added when we process i.
            # i-a edge is added when we process i-a.
            # So we just add i+a, i+b, i+558 for i in 0..n-1?
            # No, i+558 is an involution.
            
            # Let's just generate all edges and deduplicate.
            edge_set = set()
            for i in range(n_target):
                for gen in [a, n_target-a, b, n_target-b, n_target//2]:
                    j = (i + gen) % n_target
                    if i < j:
                        edge_set.add((i, j))
                    elif i > j:
                        edge_set.add((j, i))
                    # if i == j, loop, skip (but 558 is not 0, so no loop)
            
            edges = list(edge_set)
            
            # Check degree
            # Should be 5
            # Check girth
            if _verify_girth_fast(n_target, edges, g):
                return n_target, edges

        # If not found, return LUW
        n, edges = _luw_construct(5, 4)
        return n, edges

    # General fallback
    for d in range(2, 10):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    n, edges = _luw_construct(k, 10)
    return n, edges

# ... (rest of the code same as before)
