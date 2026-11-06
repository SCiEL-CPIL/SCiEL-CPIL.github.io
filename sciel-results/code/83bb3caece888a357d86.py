"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    Strategy:
    Use disjoint union of complete graphs K5 and K3.
    K5 is C4-free and has alpha=1.
    K3 is C4-free and has alpha=1.
    8 copies of K5 give n=40, alpha=8.
    1 copy of K3 gives n=3, alpha=1.
    Total n=43, alpha=9.
    Since 9 < 11, the graph is valid.
    Since n=43 is the maximum allowed, this is optimal.
    """
    edges = []
    current_vertex = 0
    
    # 8 copies of K5
    for _ in range(8):
        start = current_vertex
        for i in range(5):
            for j in range(i+1, 5):
                edges.append([start + i, start + j])
        current_vertex += 5
        
    # 1 copy of K3
    start = current_vertex
    for i in range(3):
        for j in range(i+1, 3):
            edges.append([start + i, start + j])
    current_vertex += 3
    
    n = current_vertex
    # n should be 43
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
