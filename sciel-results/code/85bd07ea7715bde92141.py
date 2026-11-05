"""Construct a C5-free graph with no independent set of size 9, maximizing n."""

def construct():
    """
    Return a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Construction: 3 disjoint copies of complement(C6 + 3K1) [9 vertices each, alpha=2]
                  2 disjoint copies of K4 [4 vertices each, alpha=1]
    Total: n = 3*9 + 2*4 = 35, alpha = 3*2 + 2*1 = 8 < 9
    C5-free since disjoint union of C5-free graphs.
    """
    edges = []
    offset = 0
    
    # Helper: add complement of (C6 + 3K1) block at given offset
    # H has vertices offset..offset+8
    # C6: vertices offset..offset+5 with cycle edges
    # 3K1: vertices offset+6, offset+7, offset+8 (isolated in H)
    # G = complement of H: all pairs not in H
    for block in range(3):
        base = offset
        # Vertices in this block: base, base+1, ..., base+8
        c6_vertices = list(range(base, base + 6))
        k1_vertices = list(range(base + 6, base + 9))
        
        # C6 edges in H (these are NOT edges in G)
        c6_edges = set()
        for i in range(6):
            u = c6_vertices[i]
            v = c6_vertices[(i + 1) % 6]
            c6_edges.add((min(u, v), max(u, v)))
        
        # All possible edges within the 9-vertex block
        all_pairs = []
        for i in range(9):
            for j in range(i + 1, 9):
                all_pairs.append((base + i, base + j))
        
        # G edges = all pairs minus C6 edges
        for u, v in all_pairs:
            if (u, v) not in c6_edges:
                edges.append([u, v])
        
        offset += 9
    
    # Add 2 copies of K4
    for block in range(2):
        base = offset
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        offset += 4
    
    n = offset  # should be 35
    return {'n': n, 'edges': edges}
