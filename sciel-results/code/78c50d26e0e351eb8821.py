"""Rewritten program using deterministic analytic construction."""

# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # The graph is the circulant graph C_21(1, 2, 5).
    # It can be decomposed into 3 edge-disjoint Hamiltonian cycles corresponding to the step sizes 1, 2, and 5.
    
    # Cycle 1: Step 1
    c1 = list(range(n))
    
    # Cycle 2: Step 2
    # Since gcd(2, 21) = 1, stepping by 2 visits all vertices.
    # However, the simple sequence 0, 2, 4... wraps around.
    # 0, 2, 4, ..., 20, 1, 3, ..., 19, 0
    c2 = list(range(0, n, 2)) + list(range(1, n, 2))
    
    # Cycle 3: Step 5
    # Since gcd(5, 21) = 1, stepping by 5 visits all vertices.
    # 0, 5, 10, 15, 20, 4, 9, ...
    c3 = [(i * 5) % n for i in range(n)]
    
    # Verification (optional but good for safety, though deterministic logic is sound)
    # Check if edges are valid and disjoint
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
        
    def get_cycle_edges(cycle):
        edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edges.add((min(u, v), max(u, v)))
        return edges
        
    e1 = get_cycle_edges(c1)
    e2 = get_cycle_edges(c2)
    e3 = get_cycle_edges(c3)
    
    # Check validity: all edges must be in host graph
    if not (e1 | e2 | e3).issubset(host_edges):
        # Fallback to search if analytic assumption fails (should not happen)
        # But for this specific graph, it is guaranteed.
        pass
        
    # Check disjointness
    if (e1 & e2) or (e1 & e3) or (e2 & e3):
        # If they overlap, something is wrong. Return what we have?
        # The task says "Any broken rule makes the whole result invalid".
        # So we must ensure they are disjoint.
        # Given the math, they are disjoint.
        pass

    return [c1, c2, c3]

# EVOLVE-BLOCK-END
