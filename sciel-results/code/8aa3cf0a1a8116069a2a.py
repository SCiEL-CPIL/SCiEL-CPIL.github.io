import random
import itertools
from functools import lru_cache

def construct():
    n = 18
    edges = []
    
    # Three parts of 6 vertices each
    # Part A: 0-5, Part B: 6-11, Part C: 12-17
    
    # Internal edges: perfect matching in each part
    # A: (0,1), (2,3), (4,5)
    # B: (6,7), (8,9), (10,11)
    # C: (12,13), (14,15), (16,17)
    for base in [0, 6, 12]:
        for i in range(3):
            edges.append([base + 2*i, base + 2*i + 1])
    
    # C4-free bipartite graphs between parts
    # Use a construction based on finite fields or known C4-free bipartite graphs
    # For 6x6, we can use a specific pattern
    
    # Between A and B: use edges (i, j) where j = i, i+1, i+2 mod 6 is NOT C4-free
    # Let's use a different approach: random C4-free bipartite graph
    
    def is_c4_free(adj_matrix, size):
        """Check if bipartite graph is C4-free"""
        for i in range(size):
            for j in range(size):
                for k in range(j+1, size):
                    for l in range(k+1, size):
                        if adj_matrix[i][j] and adj_matrix[i][k] and adj_matrix[l][j] and adj_matrix[l][k]:
                            return False
        return True
    
    # Try to construct C4-free bipartite graphs
    # Use a known construction: for n=6, use edges based on quadratic residues or similar
    
    # Let me try a specific pattern for 6x6 C4-free bipartite graph
    # One approach: use a 3-regular bipartite graph that is C4-free
    
    # Pattern for A-B: vertex i in A connects to vertices in B based on a pattern
    # Let's try: i connects to (i, i+1, i+3) mod 6
    ab_edges = []
    for i in range(6):
        for j in [i, (i+1)%6, (i+3)%6]:
            ab_edges.append((i, j))
    
    # Check if this is C4-free
    adj_ab = [[False]*6 for _ in range(6)]
    for i, j in ab_edges:
        adj_ab[i][j] = True
    
    c4_free_ab = is_c4_free(adj_ab, 6)
    
    # If not C4-free, try another pattern
    if not c4_free_ab:
        # Try: i connects to (i, i+2, i+4) mod 6
        ab_edges = []
        for i in range(6):
            for j in [i, (i+2)%6, (i+4)%6]:
                ab_edges.append((i, j))
        adj_ab = [[False]*6 for _ in range(6)]
        for i, j in ab_edges:
            adj_ab[i][j] = True
        c4_free_ab = is_c4_free(adj_ab, 6)
    
    if not c4_free_ab:
        # Try a different approach: use a 2-regular graph (cycles)
        # A 6-cycle is C4-free
        ab_edges = [(i, (i+1)%6) for i in range(6)]
        adj_ab = [[False]*6 for _ in range(6)]
        for i, j in ab_edges:
            adj_ab[i][j] = True
        c4_free_ab = is_c4_free(adj_ab, 6)
    
    # Add A-B edges
    for i, j in ab_edges:
        edges.append([i, 6+j])
    
    # Between B and C: use similar pattern
    bc_edges = []
    for i in range(6):
        for j in [i, (i+1)%6, (i+3)%6]:
            bc_edges.append((i, j))
    
    adj_bc = [[False]*6 for _ in range(6)]
    for i, j in bc_edges:
        adj_bc[i][j] = True
    
    if not is_c4_free(adj_bc, 6):
        bc_edges = [(i, (i+1)%6) for i in range(6)]
    
    for i, j in bc_edges:
        edges.append([6+i, 12+j])
    
    # Between C and A: use similar pattern
    ca_edges = []
    for i in range(6):
        for j in [i, (i+1)%6, (i+3)%6]:
            ca_edges.append((i, j))
    
    adj_ca = [[False]*6 for _ in range(6)]
    for i, j in ca_edges:
        adj_ca[i][j] = True
    
    if not is_c4_free(adj_ca, 6):
        ca_edges = [(i, (i+1)%6) for i in range(6)]
    
    for i, j in ca_edges:
        edges.append([12+i, j])
    
    # Build adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Verify K4-free
    def has_k4():
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return True
        return False
    
    if has_k4():
        # Fallback: complete tripartite K6,6,6
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Compute max triangle-free induced subgraph size using branch and bound
    def is_triangle_free(vertex_set):
        vs = list(vertex_set)
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                u, v = vs[i], vs[j]
                if not adj[u][v]:
                    continue
                for k in range(j+1, len(vs)):
                    w = vs[k]
                    if adj[u][w] and adj[v][w]:
                        return False
        return True
    
    def max_triangle_free():
        # Use branch and bound
        best = 0
        
        # Try to find a triangle-free set of size 9 first
        # If not found, try 8, then 7, etc.
        for size in range(9, 5, -1):
            for comb in itertools.combinations(range(n), size):
                if is_triangle_free(comb):
                    return size
        return 5
    
    score = max_triangle_free()
    
    return edges
