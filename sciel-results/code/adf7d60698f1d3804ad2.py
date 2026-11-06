"""R_4(C6) - 4-colouring of K_n without monochromatic C6, using bit-parallel backtracking search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for safety
    
    # We aim for n=17, then 16, etc.
    # Best known is 17.
    
    # Fallback solution: n=9 circular coloring
    def get_fallback(n):
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    best_n = 9
    best_colors = get_fallback(9)
    
    # Try to find valid colorings for decreasing n starting from 17
    # Backtracking is more likely to succeed for smaller n if 17 is too hard, 
    # but we try 17 first.
    
    for target_n in [17, 16, 15, 14]:
        if time.time() > deadline:
            break
            
        if solve_backtracking(target_n, deadline):
            best_n = target_n
            # The solution is stored in global or returned directly? 
            # Let's modify solve to return the colors
            # Actually, let's restructure to return the result
    
    # Since solve_backtracking needs to return the colors, let's wrap it properly.
    # The above loop is a placeholder. Let's implement the actual solver.
    
    # Re-implementation logic:
    
    def solve_for_n(n, time_limit):
        """
        Attempts to find a 4-coloring of K_n with no monochromatic C6.
        Returns the list of colors if found, else None.
        """
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        
        # Map edge index to (u, v)
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        # Adjacency masks for each color: adj[color][vertex] is a bitmask of neighbors
        adj = [[0] * n for _ in range(4)]
        
        # Count of edges assigned per color (optional, for symmetry breaking or stats)
        edge_count = [0] * 4
        
        # Symmetry breaking:
        # Fix colors of edges from vertex 0 to break rotational/color symmetry.
        # Color 0 is always allowed.
        # We can fix the first few edges to specific colors to reduce branching factor.
        # E.g., (0,1)=0, (0,2)=0, (0,3)=1, (0,4)=2, (0,5)=3... 
        # This is a heuristic. A standard way is to fix the first k edges of vertex 0.
        
        # Let's use a simpler symmetry break:
        # The color of edge (0,1) must be 0.
        # The color of edge (0,2) must be <= color(0,1) ? No, colors are symmetric.
        # We can fix the sequence of colors for edges (0,1), (0,2), ..., (0,k) to be non-decreasing?
        # Or just fix the first few to specific values to break color permutation symmetry.
        
        # Let's fix:
        # (0,1) = 0
        # (0,2) = 0
        # (0,3) = 1
        # (0,4) = 2
        # (0,5) = 3
        # This breaks the color symmetry significantly.
        
        initial_fixes = []
        if n >= 2: initial_fixes.append((0, 1, 0))
        if n >= 3: initial_fixes.append((0, 2, 0))
        if n >= 4: initial_fixes.append((0, 3, 1))
        if n >= 5: initial_fixes.append((0, 4, 2))
        if n >= 6: initial_fixes.append((0, 5, 3))
        
        # Apply initial fixes
        for u, v, c in initial_fixes:
            idx = edge_index(u, v, n)
            colors[idx] = c
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            edge_count[c] += 1
            # Check for C6 immediately? 
            # For n=17, initial edges don't form C6.
            
        # Order edges for backtracking
        # We should process edges in an order that maximizes constraints early.
        # Edges incident to high-degree vertices or those creating triangles might be good.
        # For simplicity, we can use lexicographic order, but maybe randomize with restarts.
        # However, for backtracking, a fixed order is better for pruning.
        # Let's stick to lexicographic but skip the fixed ones.
        
        unassigned_edges = []
        for idx in range(num_edges):
            if colors[idx] == -1:
                unassigned_edges.append(idx)
                
        # Heuristic: Process edges involving vertex 0 first?
        # Or edges that are part of many potential C6s?
        # Let's just use the natural order for now, as symmetry breaking handles the hard part.
        
        # Recursive backtracking function
        # To avoid recursion depth issues and allow time checks, we use an explicit stack or 
        # a recursive function with a counter.
        
        def has_c6_with_new_edge(c, u, v):
            """
            Check if adding edge (u,v) to color c creates a C6.
            This is equivalent to checking if there is a path of length 5 between u and v
            in the graph G_c \ {(u,v)}.
            Path: u - a - b - c - d - v
            """
            # We use bitmasks for adjacency.
            # N(x) = adj[c][x]
            
            # Iterate over 'a' in N(u)
            # a cannot be v (since we are adding u-v, v is not in N(u) yet in the *old* graph, 
            # but we are checking the graph *before* adding u-v? 
            # Actually, if we add u-v, we check if u-v completes a C6.
            # The other 5 edges must exist in the current graph.
            # So we look for paths of length 5 in the current adj[c] from u to v.
            
            # Note: adj[c] does NOT contain the edge (u,v) yet.
            
            # u - a - b - c - d - v
            # a in N(u)
            # b in N(a) \ {u}
            # c in N(b) \ {u, a}
            # d in N(c) \ {u, a, b}
            # v in N(d)
            
            # Using bitmasks:
            # mask_u = adj[c][u]
            # mask_v = adj[c][v]
            
            # For each a in mask_u:
            #   For each b in mask_a & ~mask_u:
            #     For each c in mask_b & ~(mask_u | mask_a):
            #       For each d in mask_c & ~(mask_u | mask_a | mask_b):
            #         if mask_d & mask_v: return True
            
            # This is still potentially expensive if degrees are high.
            # But n=17, max degree 16.
            
            mask_u = adj[c][u]
            mask_v = adj[c][v]
            
            # Iterate over a
            a = 0
            while mask_u:
                lsb = mask_u & -mask_u
                a = lsb.bit_length() - 1
                mask_u ^= lsb
                
                # a cannot be v? 
                # If a == v, then path is u-v... but v is the endpoint. 
                # The path is u-a-b-c-d-v. 
                # If a==v, then u-v is an edge in the path? 
                # But we are checking if u-v *completes* the cycle. 
                # The path of length 5 uses edges OTHER than (u,v).
                # So a, b, c, d must be distinct from u and v.
                
                if a == v:
                    continue
                
                mask_a = adj[c][a]
                # b in N(a), b != u, b != v (v is endpoint, not intermediate)
                # Actually, b just needs to be in N(a) and not u.
                # But b also cannot be v? 
                # If b == v, then u-a-v is a path of length 2. We need length 5.
                # So b, c, d must be distinct from u and v.
                
                candidates_b = mask_a & ~(1 << u) & ~(1 << v)
                
                b = 0
                while candidates_b:
                    lsb_b = candidates_b & -candidates_b
                    b = lsb_b.bit_length() - 1
                    candidates_b ^= lsb_b
                    
                    mask_b = adj[c][b]
                    # c in N(b), c != u, c != a, c != v
                    candidates_c = mask_b & ~(1 << u) & ~(1 << a) & ~(1 << v)
                    
                    c_node = 0
                    while candidates_c:
                        lsb_c = candidates_c & -candidates_c
                        c_node = lsb_c.bit_length() - 1
                        candidates_c ^= lsb_c
                        
                        mask_c = adj[c][c_node]
                        # d in N(c), d != u, d != a, d != b, d != v
                        candidates_d = mask_c & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                        
                        d_node = 0
                        while candidates_d:
                            lsb_d = candidates_d & -candidates_d
                            d_node = lsb_d.bit_length() - 1
                            candidates_d ^= lsb_d
                            
                            mask_d = adj[c][d_node]
                            # Check if d is connected to v
                            if mask_d & (1 << v):
                                return True
            return False

        def backtrack(idx):
            """
            idx is the index in unassigned_edges list.
            """
            if time.time() > time_limit:
                return False
            
            if idx == len(unassigned_edges):
                return True
            
            edge_idx = unassigned_edges[idx]
            u, v = edge_map[edge_idx]
            
            # Try each color
            for c in range(4):
                # Check if this color is allowed (symmetry breaking might restrict, 
                # but for now we just check C6 constraint)
                
                # Optimization: If we have fixed colors for vertex 0 edges, 
                # we might enforce order? 
                # For now, just check C6.
                
                if has_c6_with_new_edge(c, u, v):
                    continue
                
                # Assign
                colors[edge_idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                if backtrack(idx + 1):
                    return True
                
                # Unassign
                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors[edge_idx] = -1
                
            return False

        # To improve performance, we can use random restarts with different edge orderings
        # or different symmetry breaking.
        
        # For now, try the fixed symmetry breaking.
        if backtrack(0):
            return colors
        
        return None

    # Try to solve for n=17
    for n in [17, 16, 15]:
        if time.time() > deadline:
            break
            
        # Add some randomness to edge ordering to escape local traps?
        # Backtracking is deterministic. 
        # We can try multiple times with different symmetry breaks.
        
        found = False
        for attempt in range(10): # Max 10 attempts per n
            if time.time() > deadline:
                break
                
            # Reset state for each attempt? 
            # The solve_for_n function creates new state.
            res = solve_for_n(n, deadline)
            if res is not None:
                best_n = n
                best_colors = res
                found = True
                break
        
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}

def edge_index(i, j, n):
    """Get the index in the colors array for edge (i,j) with i<j in lexicographic order."""
    # Number of edges before vertex i: i*(i-1)/2
    # Within vertex i's edges, j is the (j-i-1)th edge
    return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

# EVOLVE-BLOCK-END
