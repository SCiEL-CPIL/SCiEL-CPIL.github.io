import time

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Strategy: Use 3 disjoint copies of complement of C9 and 1 copy of two disjoint K3s.
    n = 3 * 9 + 6 = 33.
    alpha = 3 * 2 + 2 = 8.
    C5-free: C9 has no C5 subgraph, so complement has no C5 subgraph. K3 has no C5.
    """
    n = 33
    edges = []
    
    # Helper to add edges for a graph defined by non-edges (complement)
    # Or just generate edges directly.
    
    # Component 1, 2, 3: Complement of C9
    # Vertices 0..8, 9..17, 18..26
    for comp in range(3):
        base = 9 * comp
        # C9 edges: (i, i+1) mod 9
        # Complement edges: all pairs EXCEPT (i, i+1) mod 9
        for i in range(9):
            for j in range(i + 1, 9):
                # Check if (i, j) is an edge in C9
                diff = abs(i - j)
                if diff == 1 or diff == 8:
                    continue
                edges.append([base + i, base + j])
                
    # Component 4: Two disjoint K3s
    # Vertices 27..32
    base = 27
    # K3 on 27, 28, 29
    edges.append([27, 28])
    edges.append([27, 29])
    edges.append([28, 29])
    # K3 on 30, 31, 32
    edges.append([30, 31])
    edges.append([30, 32])
    edges.append([31, 32])
    
    return {'n': n, 'edges': edges}
