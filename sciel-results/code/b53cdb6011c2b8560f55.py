"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
    
    # Strategy: Two-hub topology with a bridge.
    # Hubs: 1 and 23.
    # Connect 1 to left-side even vertices (excluding neighbors 0, 2).
    # Connect 23 to right-side even vertices (excluding neighbors 22, 20? No, 22 is neighbor. 20 is not).
    # Bridge: (1, 23).
    
    added_edges = []
    
    # Left hub: 1
    # Connect to evens: 4, 6, 8, 10
    for v in [4, 6, 8, 10]:
        if (1, v) not in base_edges:
            added_edges.append([1, v])
    
    # Right hub: 23
    # Connect to evens: 12, 14, 16, 18, 20
    for v in [12, 14, 16, 18, 20]:
        if (23, v) not in base_edges:
            added_edges.append([23, v])
    
    # Bridge
    if (1, 23) not in base_edges:
        added_edges.append([1, 23])
        
    return added_edges
# EVOLVE-BLOCK-END
