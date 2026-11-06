"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Precompute edge indices for n up to 20
    # edge_index[i][j] for i < j
    def get_edge_indices(n):
        """Return list of (i, j) pairs in lexicographic order."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges
    
    def build_adj(colors, n, color):
        """Build adjacency bitmasks for a given color."""
        adj = [0] * n
        for idx, c in enumerate(colors):
            if c == color:
                # Find i, j from index
                # idx = i*(i-1)/2 + (j-i-1)
                # Solve for i: i*(i-1)/2 <= idx
                i = 0
                while i * (i + 1) // 2 <= idx:
                    i += 1
                i -= 1
                j = idx - i * (i - 1) // 2 + i + 1
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj
    
    def has_c6_fast(adj, n):
        """Check if adjacency contains a 6-cycle using bit operations."""
        # For each pair (v0, v1) that are adjacent, check for a 4-path v1-v2-v3-v4-v5-v0
        # where v1<v2<v3<v4<v5 to avoid duplicates
        for v0 in range(n):
            a0 = adj[v0]
            if a0 == 0:
                continue
            # v1 must be > v0 and adjacent to v0
            v1_candidates = a0 & ~((1 << (v0 + 1)) - 1)
            while v1_candidates:
                v1_bit = v1_candidates & (-v1_candidates)
                v1 = v1_bit.bit_length() - 1
                v1_candidates ^= v1_bit
                
                a1 = adj[v1]
                # v2 > v1, adjacent to v1
                v2_candidates = a1 & ~((1 << (v1 + 1)) - 1)
                while v2_candidates:
                    v2_bit = v2_candidates & (-v2_candidates)
                    v2 = v2_bit.bit_length() - 1
                    v2_candidates ^= v2_bit
                    
                    a2 = adj[v2]
                    # v3 > v2, adjacent to v2
                    v3_candidates = a2 & ~((1 << (v2 + 1)) - 1)
                    while v3_candidates:
                        v3_bit = v3_candidates & (-v3_candidates)
                        v3 = v3_bit.bit_length() - 1
                        v3_candidates ^= v3_bit
                        
                        a3 = adj[v3]
                        # v4 > v3, adjacent to v3
                        v4_candidates = a3 & ~((1 << (v3 + 1)) - 1)
                        while v4_candidates:
                            v4_bit = v4_candidates & (-v4_candidates)
                            v4 = v4_bit.bit_length() - 1
                            v4_candidates ^= v4_bit
                            
                            a4 = adj[v4]
                            # v5 > v4, adjacent to v4, and v5 adjacent to v0
                            v5_candidates = a4 & a0 & ~((1 << (v4 + 1)) - 1)
                            if v5_candidates:
                                return True
        return False
    
    def has_any_c6(colors, n):
        """Check if any color class contains a 6-cycle."""
        for c in range(4):
            adj = build_adj(colors, n, c)
            if has_c6_fast(adj, n):
                return True
        return False
    
    def backtrack(n, colors, edge_pos, edges):
        """Backtracking search for a valid 4-coloring of K_n."""
        nonlocal deadline
        if time.time() > deadline:
            return None
        
        if edge_pos == len(edges):
            if not has_any_c6(colors, n):
                return colors[:]
            return None
        
        i, j = edges[edge_pos]
        
        # Symmetry breaking for first few edges
        if i == 0 and j == 1:
            colors[edge_pos] = 0
            result = backtrack(n, colors, edge_pos + 1, edges)
            if result is not None:
                return result
        elif i == 0 and j == 2:
            for c in range(2):
                colors[edge_pos] = c
                result = backtrack(n, colors, edge_pos + 1, edges)
                if result is not None:
                    return result
        elif i == 0 and j == 3:
            for c in range(3):
                colors[edge_pos] = c
                result = backtrack(n, colors, edge_pos + 1, edges)
                if result is not None:
                    return result
        else:
            # Try colors 0,1,2,3
            for c in range(4):
                colors[edge_pos] = c
                result = backtrack(n, colors, edge_pos + 1, edges)
                if result is not None:
                    return result
        
        colors[edge_pos] = -1
        return None
    
    # Start with the known n=9 solution
    n_start = 9
    best_n = n_start
    best_colors = [min(j - i, n_start - j + i) - 1 for i in range(n_start) for j in range(i + 1, n_start)]
    
    # Try to find larger solutions
    # For n=10, 11, ... up to 17, try backtracking
    # But this is very expensive. Let's try a smarter approach.
    
    # Actually, for n >= 10, the search space is huge. 
    # Let's try a different strategy: use a known construction if possible,
    # or use randomized search with local improvement.
    
    # For now, try backtracking for small n increments
    # But limit the search to avoid timeout
    
    # Try n=10 first with limited search
    for n_try in range(10, 18):
        if time.time() > deadline - 5:
            break
            
        edges = get_edge_indices(n_try)
        num_edges = len(edges)
        colors = [-1] * num_edges
        
        # Use backtracking with time limit
        # To make it faster, we can add pruning:
        # After placing an edge, check if any color class already has a C6
        # But this is expensive. Let's just do the full search with time limit.
        
        result = backtrack(n_try, colors, 0, edges)
        if result is not None:
            best_n = n_try
            best_colors = result
            # Don't break; try to find even larger
        else:
            # No solution found within time limit for this n
            # Try next n
            continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
