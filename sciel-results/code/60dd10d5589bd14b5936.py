"""
4-colourings of complete graphs without monochromatic C6.
Uses incremental backtracking with fast C6 detection via bitmask adjacency.
"""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Try to find solutions for n from 17 down to 10
    # We prioritize higher n.
    
    best_n = 9
    # Known n=9 solution: cyclic coloring
    n0 = 9
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    for target_n in range(17, 9, -1):
        if time.time() > deadline - 15:
            break
        
        # Try to find a valid coloring for target_n
        solution = _solve_for_n(target_n, deadline)
        if solution is not None:
            best_n = target_n
            best_colors = solution
            break
    
    return {'n': best_n, 'colors': best_colors}


def _solve_for_n(n, deadline):
    """
    Attempt to find a 4-coloring of K_n with no monochromatic C6.
    Uses incremental backtracking: add vertices one by one, assigning colors
    to edges from new vertex to existing vertices, checking for C6 violations.
    Returns list of colors or None if failed.
    """
    # We'll build the coloring incrementally.
    # colors[i][j] for i < j stores the color.
    # adj[c][i] is a bitmask of neighbors of i in color c.
    
    adj = [[0] * n for _ in range(4)]  # adj[c][v] = bitmask of neighbors in color c
    colors_matrix = [[-1] * n for _ in range(n)]  # colors_matrix[i][j] = color of edge (i,j)
    
    # Precompute edge index mapping
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Fast C6 check: given a color c and a bitmask of candidate vertices,
    # check if there's a C6 in the subgraph induced by those vertices in color c.
    # For incremental construction, we check if adding new edges creates a C6.
    
    def _has_c6_in_color(adj_c):
        """Check if color class adj_c contains a C6."""
        # For each vertex u, check if there's a path of length 5 from u back to u
        # with all intermediate vertices distinct and > u (to avoid duplicates).
        for u in range(n):
            # Neighbors of u in color c, greater than u
            N_u = adj_c[u] & ~((1 << (u + 1)) - 1)
            if N_u == 0:
                continue
            
            # DFS to find path of length 5 from u
            # Path: u -> v1 -> v2 -> v3 -> v4 -> v5 -> u
            # v1, v2, v3, v4, v5 all > u and distinct
            
            def dfs(current, depth, visited_mask):
                if depth == 5:
                    # Check if current is connected to u
                    return (adj_c[current] >> u) & 1
                # Next vertices must be > u and not visited
                candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~visited_mask
                c_bits = candidates
                while c_bits:
                    c_bit = c_bits & -c_bits
                    c = c_bit.bit_length() - 1
                    c_bits ^= c_bit
                    if dfs(c, depth + 1, visited_mask | c_bit):
                        return True
                return False
            
            s_bits = N_u
            while s_bits:
                s_bit = s_bits & -s_bits
                s = s_bit.bit_length() - 1
                s_bits ^= s_bit
                if dfs(s, 1, s_bit):
                    return True
        return False
    
    # For incremental construction, we need a faster check.
    # When adding vertex v with edges to vertices 0..v-1, we only need to check
    # if any C6 is formed that involves at least one of the new edges.
    # A C6 involving new vertex v must have v in the cycle.
    # So we check: for each color c, is there a C6 in the subgraph on {0..v} 
    # that uses at least one edge incident to v?
    
    def _has_c6_involving_v(adj_c, v):
        """Check if there's a C6 in color c that includes vertex v."""
        # C6 containing v: v - a - b - c - d - e - v
        # where a, b, c, d, e are distinct and not v.
        # We can fix v and look for paths of length 5 from v back to v.
        # But we need to be careful: the path must use vertices from 0..v-1 (existing) 
        # and possibly v itself only at start/end.
        
        # Neighbors of v in color c (among vertices 0..v-1)
        N_v = adj_c[v] & ((1 << v) - 1)
        if N_v == 0:
            return False
        
        # For each neighbor a of v, look for path a - b - c - d - e - v
        # where b, c, d, e are distinct, not v, not a, and in 0..v-1.
        # This is a path of length 4 from a to v in G_c - {v} (but v is endpoint).
        
        a_bits = N_v
        while a_bits:
            a_bit = a_bits & -a_bits
            a = a_bit.bit_length() - 1
            a_bits ^= a_bit
            
            # From a, find path of length 4 to v: a - b - c - d - e - v
            # b is neighbor of a in color c, b != v, b != a
            # c is neighbor of b in color c, c != v, c != a, c != b
            # d is neighbor of c in color c, d != v, d != a, d != b, d != c
            # e is neighbor of d in color c, e != v, e != a, e != b, e != c, e != d
            # and e is connected to v in color c.
            
            # DFS from a, avoiding v, looking for path of length 4 ending at neighbor of v
            def dfs_from_a(current, depth, visited_mask):
                if depth == 4:
                    # current should be connected to v
                    return (adj_c[current] >> v) & 1
                # Next vertices: neighbors of current in color c, not v, not visited
                candidates = adj_c[current] & ((1 << v) - 1) & ~visited_mask
                cand_bits = candidates
                while cand_bits:
                    cand_bit = cand_bits & -cand_bits
                    cand = cand_bit.bit_length() - 1
                    cand_bits ^= cand_bit
                    if dfs_from_a(cand, depth + 1, visited_mask | cand_bit):
                        return True
                return False
            
            if dfs_from_a(a, 1, a_bit):
                return True
        return False
    
    # Incremental backtracking
    # State: for vertex v (0-indexed), we assign colors to edges (v, 0), (v, 1), ..., (v, v-1)
    # We use recursion with backtracking.
    
    # To make it faster, we order the color assignments and prune early.
    
    # For each new vertex v, we need to assign colors to v edges.
    # We try all 4^v combinations? No, that's too many for v > 10.
    # Instead, we assign one edge at a time and check for C6 violations incrementally.
    
    # Better approach: for vertex v, assign colors to edges (v, 0), (v, 1), ..., (v, v-1)
    # one at a time. After each assignment, check if any C6 is formed.
    # If a C6 is formed, backtrack.
    
    # But checking for C6 after each single edge assignment is expensive.
    # Alternative: assign all v edges, then check once. But 4^v is large.
    
    # Compromise: use randomized search with local repair.
    # Initialize with random coloring, then fix violations.
    
    # Let's try a hybrid: random initialization + local search with efficient C6 detection.
    
    num_edges = n * (n - 1) // 2
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    
    # Build adjacency
    def build_adj():
        for c in range(4):
            for i in range(n):
                adj[c][i] = 0
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
    
    build_adj()
    
    # Precompute edge list for quick lookup
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    def find_c6_and_edge():
        """Find a monochromatic C6 and return (color, edge_idx_in_cycle)."""
        for c in range(4):
            adj_c = adj[c]
            for u in range(n):
                N_u = adj_c[u] & ~((1 << (u + 1)) - 1)
                if N_u == 0:
                    continue
                
                def dfs(current, depth, path):
                    if depth == 5:
                        if (adj_c[current] >> u) & 1:
                            return path + [current]
                        return None
                    candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~((1 << (depth)) - 1) if depth == 1 else adj_c[current] & ~((1 << (u + 1)) - 1)
                    # Build visited mask from path
                    visited = 0
                    for p in path:
                        visited |= (1 << p)
                    candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~visited
                    c_bits = candidates
                    while c_bits:
                        c_bit = c_bits & -c_bits
                        cv = c_bit.bit_length() - 1
                        c_bits ^= c_bit
                        result = dfs(cv, depth + 1, path + [cv])
                        if result is not None:
                            return result
                    return None
                
                s_bits = N_u
                while s_bits:
                    s_bit = s_bits & -s_bits
                    s = s_bit.bit_length() - 1
                    s_bits ^= s_bit
                    result = dfs(s, 1, [s])
                    if result is not None:
                        # Cycle: u, result[0], result[1], result[2], result[3], result[4], u
                        cycle = [u] + result
                        # Find an edge in the cycle to flip
                        # Edge indices in cycle: (u, result[0]), (result[0], result[1]), ..., (result[4], u)
                        edge_pairs = [(cycle[i], cycle[(i + 1) % 6]) for i in range(6)]
                        # Pick a random edge
                        k = random.randint(0, 5)
                        i, j = edge_pairs[k]
                        if i > j:
                            i, j = j, i
                        idx = edge_idx(i, j)
                        return c, idx
        return None, None
    
    max_steps = 50000
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            return None
        
        # Check if current coloring is valid
        c, idx = find_c6_and_edge()
        if c is None:
            return colors
        
        # Flip the edge at idx
        old_c = colors[idx]
        new_c = random.choice([cc for cc in range(4) if cc != old_c])
        colors[idx] = new_c
        
        # Update adjacency
        u, v = edges[idx]
        adj[old_c][u] &= ~(1 << v)
        adj[old_c][v] &= ~(1 << u)
        adj[new_c][u] |= (1 << v)
        adj[new_c][v] |= (1 << u)
        
        step += 1
    
    return None
# EVOLVE-BLOCK-END
