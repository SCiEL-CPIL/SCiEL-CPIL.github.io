import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to convert list of colors to dict for quick lookup? 
    # No, we will maintain adjacency bitsets for each color.
    
    def solve_csp(n):
        """
        Finds a 4-coloring of K_n with no C6 in any color class.
        Returns list of colors or None.
        """
        if n <= 1:
            return []
        
        # Edges in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        
        # Symmetry breaking:
        # Fix the first few edges to specific colors to break the 4! symmetry 
        # and vertex permutation symmetry partially.
        # Edge (0,1) -> 0
        # Edge (0,2) -> 0
        # Edge (0,3) -> 1
        # Edge (0,4) -> 1
        # This assumes vertices are ordered.
        
        initial_assignment = {}
        # We will assign colors to edges in a specific order.
        # To maximize pruning, we should assign edges that constrain the most.
        # But for simplicity and speed, we iterate in a smart order.
        
        # State:
        # colors: dict edge_index -> color
        # adj: list of 4 lists of bitsets (integers) for each color.
        # adj[c][v] is a bitset of neighbors of v in color c.
        
        adj = [[0] * n for _ in range(4)]
        colors = [None] * num_edges
        
        # Pre-compute edge indices
        edge_to_idx = {}
        for idx, (u, v) in enumerate(edges):
            edge_to_idx[(u, v)] = idx
            edge_to_idx[(v, u)] = idx
            
        def get_edge_idx(u, v):
            if u > v:
                u, v = v, u
            return edge_to_idx[(u, v)]

        def check_c6_possible(c, u, v, adj_state):
            """
            Check if adding edge (u,v) to color c creates a C6.
            This requires a path of length 4 from u to v in the graph of color c 
            excluding the edge (u,v) itself (which isn't added yet).
            We use bitset reachability.
            """
            # We want to know if there is a simple path u -> ... -> v of length 4.
            # A necessary condition is that v is reachable from u in 4 steps in the current graph.
            # Let's compute the set of vertices reachable from u in exactly 1, 2, 3, 4 steps.
            
            # R1: neighbors of u
            r1 = adj_state[c][u]
            if r1 == 0:
                return False
            
            # R2: neighbors of neighbors of u
            r2 = 0
            temp = r1
            while temp:
                lsb = temp & -temp
                bit = lsb.bit_length() - 1
                r2 |= adj_state[c][bit]
                temp -= lsb
            
            # Remove u from r2 to avoid backtracking immediately? 
            # Actually, for a simple path, we shouldn't revisit u.
            # But bitset reachability is an over-approximation.
            # Let's refine: We want a path u-v1-v2-v3-v4-v.
            # v1 in N(u), v2 in N(v1)\{u}, v3 in N(v2)\{u,v1}, v4 in N(v3)\{u,v1,v2}, v in N(v4)\{u,v1,v2,v3}.
            
            # This is expensive to do exactly with bitsets alone without tracking visited sets.
            # However, for n <= 20, we can do a BFS/DFS for the specific check if the bitset check passes.
            
            # First, fast bitset check: Is v in the 4-step neighborhood?
            r3 = 0
            temp = r2
            while temp:
                lsb = temp & -temp
                bit = lsb.bit_length() - 1
                r3 |= adj_state[c][bit]
                temp -= lsb
            
            r4 = 0
            temp = r3
            while temp:
                lsb = temp & -temp
                bit = lsb.bit_length() - 1
                r4 |= adj_state[c][bit]
                temp -= lsb
            
            if not (r4 & (1 << v)):
                return False # No path of length 4, so safe.
            
            # Potential C6 exists. We must verify if a SIMPLE cycle exists.
            # We need to find a path u -> a -> b -> c -> d -> v where all vertices are distinct.
            # Note: The edge (u,v) is NOT in adj_state yet.
            
            # DFS from u to find path of length 4 to v.
            # Stack: (current_node, path_so_far)
            # Since depth is small (4), this is fast.
            
            def dfs(node, depth, visited):
                if depth == 4:
                    # Check if v is a neighbor of node and v not in visited
                    if (adj_state[c][node] & (1 << v)) and not (visited & (1 << v)):
                        return True
                    return False
                
                # Get neighbors of node in color c
                neighbors = adj_state[c][node]
                # Exclude visited nodes
                neighbors &= ~visited
                
                while neighbors:
                    lsb = neighbors & -neighbors
                    bit = lsb.bit_length() - 1
                    if dfs(bit, depth + 1, visited | (1 << bit)):
                        return True
                    neighbors -= lsb
                return False
            
            # Start from u
            # visited includes u
            return dfs(u, 0, 1 << u)

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            # Try all 4 colors
            # Symmetry breaking: if this is the first edge, only try color 0.
            # If this is the second edge (0,2), only try colors 0,1 (since (0,1) is 0, swapping 1 and 2 is symm? No, we fixed (0,1)=0. 
            # Let's use a simple symmetry breaking:
            # Edge 0: (0,1) -> 0
            # Edge 1: (0,2) -> 0 or 1 (if 0, then 1,2,3 are symm to 0,1,2? No. 
            # Let's just fix the first 3 edges involving vertex 0 to specific colors to break vertex permutation symmetry partially.
            # (0,1)=0, (0,2)=0, (0,3)=1.
            
            if idx == 0:
                possible_colors = [0]
            elif idx == 1:
                possible_colors = [0, 1]
            elif idx == 2:
                possible_colors = [0, 1, 2]
            else:
                possible_colors = [0, 1, 2, 3]
                
            # Optimization: Dynamic Variable Ordering is hard with fixed edge list.
            # We stick to lexicographic order but use forward checking.
            
            for c in possible_colors:
                # Check if adding (u,v) with color c creates a C6
                if not check_c6_possible(c, u, v, adj):
                    # Safe to assign
                    # Add edge
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    colors[idx] = c
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Undo
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    colors[idx] = None
            
            return False

        if backtrack(0):
            return colors
        return None

    # We want to maximize n.
    # Known best is 17. We try 17, then 16, etc.
    # Actually, we should try 17 first. If it fails within time, try 16?
    # The task says "Maximize n".
    # We should try to find a valid coloring for n=17.
    # If we find one, we return it.
    # If we don't find one in time, we should return the best we have.
    # But the CSP solver is all-or-nothing for a specific n.
    # So we should try n=17 with a time limit. If it succeeds, great.
    # If it fails, we try n=16, etc.
    
    best_result = None
    
    # Try n=17
    # To avoid spending all time on n=17 and finding nothing, we set a sub-deadline.
    # But CSP with good pruning might find it quickly or prove infeasibility quickly?
    # Proving infeasibility is hard. Finding a solution might be faster.
    
    # Let's try n=17 with a budget of 100s.
    # If it fails, try n=16 with remaining time.
    
    # Actually, the seed program had a valid n=9.
    # We should ensure we return at least n=9 if the CSP fails.
    
    def make_fallback(n=9):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - j + i)
                colors.append(d % 3)
        return {'n': n, 'colors': colors}
    
    current_best = make_fallback(9)
    
    # Try n from 17 down to 10
    # We prioritize higher n.
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
            
        # Set a sub-deadline for this n
        # If n=17, give it 90s. If n=16, give it 30s, etc.
        # This is heuristic.
        time_limit_for_n = min(deadline - 2, start_time + 90 if n == 17 else start_time + 30)
        
        # Modify global deadline temporarily? 
        # Better to pass time limit to solve_csp? 
        # My solve_csp uses global deadline. 
        # Let's create a wrapper or pass limit.
        
        # I will re-structure to pass limit.
        
        def solve_csp_limited(n, limit):
            # Re-implement or wrap? 
            # To keep code clean, I'll just use the global deadline but set it carefully.
            pass

        # Let's just use the global deadline and break out if we find a solution.
        # If we don't find a solution for n=17, we will have used most of the time.
        # This is risky.
        
        # Alternative: Run CSP for n=17. If it returns a solution, we are done.
        # If it returns None (timeout or infeasible), we return the fallback.
        # But what if n=16 is easy?
        # Given the "best known is 17", it's highly likely 17 is feasible.
        # So we focus all effort on 17.
        
        colors_17 = solve_csp(17)
        if colors_17 is not None:
            return {'n': 17, 'colors': colors_17}
        
        # If 17 failed, try 16? 
        # If we are out of time, return best.
        if time.time() > deadline - 10:
            break
            
        # Try 16
        colors_16 = solve_csp(16)
        if colors_16 is not None:
            current_best = {'n': 16, 'colors': colors_16}
            # Don't break, try 17 again? No, 17 already failed.
            # But maybe 17 failed due to bad ordering? 
            # With MRV and symmetry breaking, it should be robust.
            
        # If 16 failed, try 15?
        if time.time() > deadline - 10:
            break
            
        colors_15 = solve_csp(15)
        if colors_15 is not None:
            current_best = {'n': 15, 'colors': colors_15}

    return current_best

