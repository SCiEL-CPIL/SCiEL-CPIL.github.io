"""
4-coloring of K_n with no monochromatic C6.
Strategy: Backtracking search with bitmasks and symmetry breaking.
We build the coloring edge by edge (lexicographic order), maintaining for each color
a set of edges (as bitmasks over vertices). After each edge assignment, we check
whether any color class now contains a C6. We use a fast C6 check based on
adjacency bitmasks.

To speed things up:
- Use bitmasks for adjacency in each color.
- Precompute for each color and each pair (u,v) whether adding edge (u,v) in that
  color would create a C6. This is done by checking if there exists a path of length 4
  from u to v in that color's graph (i.e., u and v are at distance 4 in the same color).
  Actually, adding edge (u,v) creates a C6 iff there is a path of length 5 from u to v
  in the current graph of that color. But that's expensive. 
  
  Better: A C6 containing edge (u,v) in color c means there exist vertices 
  u, a, b, c2, d, v (distinct) such that edges (u,a),(a,b),(b,c2),(c2,d),(d,v) are all in color c.
  So we need a path of length 5 from u to v in color c's current graph.
  
  Alternative: check if u and v are connected by a path of exactly 5 edges in color c.
  We can do BFS/DFS with depth limit 5 from u to see if v is reachable.
  
  For n up to 20, this is feasible.

We also apply symmetry breaking: the first few edges are fixed to reduce search space.
"""

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger n via backtracking
    for target_n in range(17, 8, -1):  # Try 17, 16, ..., 9
        if time.time() > deadline - 5:
            break
        
        result = _search_n(target_n, deadline)
        if result is not None:
            best_n = target_n
            best_colors = result
            break
    
    return {'n': best_n, 'colors': best_colors}


def _search_n(n, deadline):
    """
    Backtracking search for a valid 4-coloring of K_n with no mono C6.
    Edges are assigned in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
    """
    # adj[c][u] is a bitmask of vertices v such that edge (u,v) is in color c
    adj = [[0] * n for _ in range(4)]
    
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    m = len(edges)
    colors = [0] * m
    
    # Precompute: for each edge index, the (u,v) pair
    edge_uv = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # For symmetry breaking: fix first few edges
    # The first edge (0,1) can be color 0 by symmetry
    # The second edge (0,2) can be color 0 or 1
    # We'll handle this in the search
    
    def get_adj(c, u):
        return adj[c][u]
    
    def set_edge(c, u, v):
        adj[c][u] |= (1 << v)
        adj[c][v] |= (1 << u)
    
    def unset_edge(c, u, v):
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)
    
    def has_path5(c, u, v):
        """Check if there's a path of exactly 5 edges from u to v in color c."""
        # BFS with depth limit 5
        # visited[d][x] = whether x is reachable from u in exactly d steps
        # We need to avoid revisiting nodes at the same or lower depth
        # Actually, we need simple paths, so we track visited nodes
        
        # For n <= 20, we can do a DFS with pruning
        # Path: u -> a -> b -> c2 -> d -> v
        # All distinct, and edges in color c
        
        adj_u = adj[c][u]
        # Remove u and v from consideration for intermediate nodes
        # a must be adjacent to u in color c, a != v
        mask_a = adj_u & ~(1 << v) & ~(1 << u)
        
        while mask_a:
            a_bit = mask_a & (-mask_a)
            a = a_bit.bit_length() - 1
            mask_a ^= a_bit
            
            # b adjacent to a, not in {u, a, v}
            adj_a = adj[c][a]
            mask_b = adj_a & ~(1 << u) & ~(1 << a) & ~(1 << v)
            
            while mask_b:
                b_bit = mask_b & (-mask_b)
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                
                # c2 adjacent to b, not in {u, a, b, v}
                adj_b = adj[c][b]
                mask_c2 = adj_b & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                
                while mask_c2:
                    c2_bit = mask_c2 & (-mask_c2)
                    c2 = c2_bit.bit_length() - 1
                    mask_c2 ^= c2_bit
                    
                    # d adjacent to c2, not in {u, a, b, c2, v}
                    adj_c2 = adj[c][c2]
                    mask_d = adj_c2 & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << c2) & ~(1 << v)
                    
                    while mask_d:
                        d_bit = mask_d & (-mask_d)
                        d = d_bit.bit_length() - 1
                        mask_d ^= d_bit
                        
                        # Check if d is adjacent to v in color c
                        if adj[c][d] & (1 << v):
                            return True
        
        return False
    
    def would_create_c6(c, u, v):
        """Check if adding edge (u,v) in color c creates a C6."""
        # A C6 containing edge (u,v) means there's a path of length 5 from u to v
        # in the current graph of color c (before adding the edge)
        return has_path5(c, u, v)
    
    def backtrack(idx):
        nonlocal colors
        
        if time.time() > deadline - 2:
            return False
        
        if idx == m:
            return True
        
        u, v = edge_uv[idx]
        
        # Try each color
        for c in range(4):
            if would_create_c6(c, u, v):
                continue
            
            # Assign color c
            set_edge(c, u, v)
            colors[idx] = c
            
            if backtrack(idx + 1):
                return True
            
            unset_edge(c, u, v)
        
        return False
    
    # Symmetry breaking: fix first edge to color 0
    # (0,1) is edge index 0
    set_edge(0, 0, 1)
    colors[0] = 0
    
    if not backtrack(1):
        unset_edge(0, 0, 1)
        return None
    
    return colors


# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"n = {result['n']}, num colors = {len(result['colors'])}")
