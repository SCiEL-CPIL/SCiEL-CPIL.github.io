import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # Known good graph: K3,3 disjoint union K3,3
    # Vertices 0-5 and 6-11
    # K3,3 edges:
    # (0,3), (0,4), (0,5)
    # (1,3), (1,4), (1,5)
    # (2,3), (2,4), (2,5)
    # And same for 6-11
    
    best_edges = None
    best_score = float('inf')
    
    # Define candidate graphs
    # 1. Two K3,3
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                u = offset + i
                v = offset + 3 + j
                edges.append([u, v])
        return edges
    
    edges_1 = make_k33(0) + make_k33(6)
    best_edges = edges_1
    best_score = 2 # K3,3 has 4 and 6
    
    # 2. Try to find a graph with 1 length via random search?
    # Too slow for 50s to guarantee.
    
    # Check if we can do better with a specific structure
    # What about the Dodecahedron? 20 vertices. No.
    
    # Return the best found
    return [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
# EVOLVE-BLOCK-END
