import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s limit, keep 35s buffer)
    
    # Fallback solution: n=11 (known valid)
    # K11 split into five Hamiltonian cycles by circular distance 1..5
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    # Strategy: Greedy Backtracking with Bitmask Constraint Propagation
    # We build the graph vertex by vertex.
    # State:
    #   colors: list of length n*(n-1)/2, -1 if unassigned
    #   adj_mask[c][u]: bitmask of neighbors of u in color c
    #   edge_color[i][j]: color of edge (i,j), -1 if unassigned
    
    def solve_n(n, time_limit, seed):
        """
        Attempts to find a valid 5-coloring of K_n using backtracking.
        Returns colors list or None.
        """
        if time.time() > time_limit:
            return None
            
        random.seed(seed)
        
        num_edges = n * (n - 1) // 2
        # Precompute edge indices for fast lookup
        # edge_idx[i][j] = index in the flat colors array
        edge_idx = [[-1] * n for _ in range(n)]
        edge_list = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[i][j] = idx
                edge_idx[j][i] = idx
                edge_list.append((i, j))
                idx += 1
        
        # State
        colors = [-1] * num_edges
        # adj_mask[c][u] is an integer bitmask
        adj_mask = [[0] * n for _ in range(5)]
        
        # Recursion limit
        sys.setrecursionlimit(10000)
        
        # Order of vertices to process: We add vertices 0 to n-1.
        # When adding vertex k, we assign colors to edges (k, 0), (k, 1), ..., (k, k-1).
        # This is a "vertex-by-vertex" construction.
        
        def backtrack(k):
            """
            Assign colors for edges connecting vertex k to vertices 0..k-1.
            Returns True if a valid coloring is found for the whole graph (when k == n).
            """
            if time.time() > time_limit:
                return False
                
            if k == n:
                return True
                
            # We need to assign colors for edges (k, 0), (k, 1), ..., (k, k-1).
            # To speed up, we can permute the order in which we assign these edges?
            # Or just assign them in order 0..k-1.
            # Let's try to assign in a random order or a specific heuristic order.
            # Actually, the constraints are local. 
            # When assigning (k, j), we check if adding this edge creates a C4.
            # A C4 is formed if there exists w < k such that:
            #   (k, w) is color c AND (j, w) is color c.
            # Wait, the C4 condition is: no two vertices have two common neighbors in one color.
            # If we assign (k, j) = c, we must ensure that k and j do not share 2 common neighbors in color c.
            # The common neighbors of k and j in color c are vertices w such that (k, w)=c and (j, w)=c.
            # Since we are building vertex by vertex, all neighbors of k so far are < k.
            # All neighbors of j so far are < k (since j < k).
            # So we just need to count how many w < k satisfy (k, w)=c and (j, w)=c.
            
            # To make this efficient, we can maintain for each color c, the set of neighbors.
            # But we can also just compute the intersection on the fly if k is small.
            # However, k goes up to 29. 2^29 is large for bitmasks, but Python ints handle it.
            # The check:
            #   common = adj_mask[c][k] & adj_mask[c][j]
            #   if common.bit_count() >= 2: invalid.
            
            # But wait, adj_mask[c][k] is being built.
            # When we assign (k, j), we update adj_mask[c][k] and adj_mask[c][j].
            # Before assignment, adj_mask[c][k] contains neighbors of k among 0..k-1.
            # adj_mask[c][j] contains neighbors of j among 0..j-1 (and potentially some > j if we had assigned them, but we haven't assigned edges from j to > j yet? 
            # No, in vertex-by-vertex, when we are at vertex k, all edges among 0..k-1 are already assigned.
            # So adj_mask[c][j] is complete for neighbors in 0..k-1.
            # adj_mask[c][k] is partial, containing neighbors assigned so far in this step.
            
            # So the check is:
            #   potential_common = adj_mask[c][k] & adj_mask[c][j]
            #   if potential_common.bit_count() >= 2: invalid.
            
            # This is O(1) with bitmasks!
            
            # We need to assign colors to (k, 0), (k, 1), ..., (k, k-1).
            # Let's use a recursive step for each edge? 
            # Or a loop?
            # If we use a loop, we need to backtrack within the loop if a choice fails.
            # This requires a nested recursion or a stack.
            # Let's use a helper function that assigns the m-th edge of vertex k.
            
            def assign_edge(m):
                """
                Assign color to edge (k, m), where m < k.
                m goes from 0 to k-1.
                """
                if time.time() > time_limit:
                    return False
                
                if m == k:
                    # All edges for vertex k assigned. Recurse to next vertex.
                    return backtrack(k + 1)
                
                # Try colors 0..4
                # Heuristic: Try colors that are less frequent? Or random?
                # Random is good for diversification with restarts.
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    # Check constraint
                    # common neighbors of k and m in color c
                    # adj_mask[c][k] has bits for neighbors of k assigned so far (subset of 0..m-1)
                    # adj_mask[c][m] has bits for neighbors of m in 0..k-1
                    common = adj_mask[c][k] & adj_mask[c][m]
                    if common.bit_count() >= 2:
                        continue
                    
                    # Assign
                    idx = edge_idx[k][m]
                    colors[idx] = c
                    adj_mask[c][k] |= (1 << m)
                    adj_mask[c][m] |= (1 << k)
                    
                    if assign_edge(m + 1):
                        return True
                    
                    # Undo
                    colors[idx] = -1
                    adj_mask[c][k] &= ~(1 << m)
                    adj_mask[c][m] &= ~(1 << k)
                
                return False
            
            return assign_edge(0)
        
        # Start backtracking from vertex 0 (no edges to assign)
        # Actually, vertex 0 has no edges to 0..-1. So backtrack(0) calls assign_edge(0) which immediately returns backtrack(1).
        # So we can just call backtrack(0).
        
        # To increase chances, we can try different vertex orderings?
        # The standard 0..n-1 is fixed. But we can randomize the *color* choices.
        # Also, we can add a "time slice" to restart if stuck.
        
        # Let's run multiple restarts with different seeds
        # But the function `solve_n` is called once per n.
        # We should handle restarts in the main loop.
        
        # However, backtracking can be very slow if it goes deep and fails.
        # Let's add a node count limit? Or just rely on time limit.
        
        return backtrack(0)

    # Main loop: Try to find larger n
    # We know n=11 is valid.
    # Let's try n=12, 13, ... up to 26 (or until time runs out).
    
    current_seed = 42
    for n in range(12, 27):
        if time.time() > deadline - 2.0:
            break
            
        # Try a few restarts for each n
        found = False
        # Limit restarts to avoid wasting time on impossible n
        # n=12 is easy, n=26 is hard.
        # Let's give more time to smaller n? No, we want max n.
        # If n=12 fails, n=13 likely fails.
        # So we should break if n fails?
        # Not necessarily, random seed might work.
        # But if backtracking is deterministic given seed, we need different seeds.
        
        max_restarts = 20 if n < 15 else 5
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
            
            seed = current_seed + r * 1000 + n
            colors = solve_n(n, deadline - 2.0, seed)
            
            if colors is not None:
                # Verify just in case (optional, but good for safety)
                # The solver should guarantee validity.
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            # Found n, try n+1
            continue
        else:
            # Failed to find n. 
            # Is it worth trying n+1? Probably not, but maybe n was just unlucky?
            # Given the time budget, if n=12 fails with 20 restarts, it's likely hard.
            # But let's be conservative: if n <= 15, try again with more restarts?
            # Or just break?
            # The task says "Maximize n".
            # If we can't find 12, we return 11.
            # If we find 12 but not 13, we return 12.
            # Breaking is a reasonable heuristic to save time for verification or smaller n if we skipped.
            # But we are iterating upwards.
            break

    return {'n': best_n, 'colors': best_colors}
