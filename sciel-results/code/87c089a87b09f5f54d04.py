"""
Triangle-free graph with alpha < 10, maximizing n for R(3,10).

Strategy: Construct the lexicographic product C5[C8].
- C5 is the 5-cycle, triangle-free with alpha(C5) = 2.
- C8 is the 8-cycle, triangle-free with alpha(C8) = 4.
- The lexicographic product H[G] is triangle-free if both H and G are triangle-free.
- alpha(C5[C8]) = alpha(C5) * alpha(C8) = 2 * 4 = 8 < 10.
- n = 5 * 8 = 40, which is the maximum possible since R(3,10) <= 41.

Vertex ID: i * 8 + j for i in 0..4 (C5) and j in 0..7 (C8).
Edges:
1. Within the same copy i: (i, j) ~ (i, k) if j ~ k in C8.
2. Between different copies i and i': if i ~ i' in C5, then (i, j) ~ (i', k) for all j, k.
"""

def construct():
    n = 40
    edges = []
    
    # C5 adjacency: i ~ (i+1) % 5
    c5_adj = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3},
        3: {2, 4},
        4: {3, 0}
    }
    
    # C8 adjacency: j ~ (j+1) % 8
    c8_adj = {
        j: {(j + 1) % 8, (j - 1) % 8} for j in range(8)
    }
    
    for i in range(5):
        for j in range(8):
            u = i * 8 + j
            # Within the same C5 copy: connect if j ~ k in C8
            for k in c8_adj[j]:
                v = i * 8 + k
                if u < v:
                    edges.append([u, v])
            # Between different C5 copies: if i ~ i' in C5, connect all j to all k
            for ip in c5_adj[i]:
                if ip > i:  # avoid duplicate edges
                    for k in range(8):
                        v = ip * 8 + k
                        if u < v:
                            edges.append([u, v])
    
    return {'n': n, 'edges': edges}
