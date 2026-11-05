import time
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    # We'll try to solve for n=26 first (known achievable via Lazebnik-Woldar)
    # If that fails within time, fall back to smaller n

    def try_solve_n(n, num_colors=5, time_budget=None):
        """Local search with incremental penalty update for n vertices."""
        if time_budget is None:
            time_budget = deadline - time.time()
        
        # Edge list in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        m = len(edges)
        
        # color[edge_idx] gives the color
        col = np.zeros(m, dtype=np.int8)
        for k in range(m):
            col[k] = random.randint(0, num_colors - 1)
        
        # Build adjacency bitsets: adj[c][v] = bitmask of neighbors of v in color c
        # Using list of Python ints for speed with n<=29
        adj = [[0] * n for _ in range(num_colors)]
        
        # Build initial adjacency
        for k, (i, j) in enumerate(edges):
            c = col[k]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # Precompute for each pair (u,v), the list of edge indices where one endpoint is u and other is v
        # Actually, we need: for a given edge (i,j) with color c, which pairs (u,v) have i and j as common neighbors in color c?
        # When we change edge (i,j) from old_c to new_c:
        #   - Remove: i is no longer a neighbor of j in old_c, j no longer neighbor of i in old_c
        #   - Add: i is now neighbor of j in new_c, j now neighbor of i in new_c
        #
        # The penalty change: for each color c, C4 count = sum over pairs {u,v} of C(common_neigh_c(u,v), 2)
        # When we remove edge (i,j) from old_c:
        #   For any pair (u,v) where u != i,j and v != i,j:
        #     If u was adjacent to both i and j in old_c, and v was adjacent to both i and j in old_c... 
        #     Actually no. The common neighbors of u and v in color c changes only if one of u,v is i or j.
        #
        # Let me think more carefully. The C4 count for color c is:
        #   sum_{u < v} C(|N_c(u) ∩ N_c(v)|, 2)
        #
        # When we remove edge (i,j) from color old_c:
        #   N_oldc(i) loses j, N_oldc(j) loses i
        #   For pair (i, v) where v != j: common = N_oldc(i) ∩ N_oldc(v) changes (loses j if j was in N_oldc(v))
        #   For pair (j, v) where v != i: common = N_oldc(j) ∩ N_oldc(v) changes (loses i if i was in N_oldc(v))
        #   For pair (i, j): common = N_oldc(i) ∩ N_oldc(j) changes
        #   For pair (u, v) where u,v != i,j: no change
        #
        # So the penalty delta when removing edge (i,j) from color c:
        #   delta_remove = -[C(|Nc(i)∩Nc(j)|,2) + sum_{v≠i,j} (change in C for pair (i,v)) + (change for pair (j,v))]
        #
        # For pair (i,v): old common size = |Nc(i) ∩ Nc(v)|, new common size = |Nc(i) ∩ Nc(v)| - [1 if j ∈ Nc(v)]
        #   change in C = C(k,2) - C(k - [j∈Nc(v)], 2) where k = |Nc(i)∩Nc(v)|
        #
        # This is getting complex. Let me use a different approach:
        # For each color c, maintain a count of C4s. When we flip an edge, only O(n) pairs are affected.
        
        # Let me define a function to compute the penalty contribution of a single color c
        # Given adj[c], the penalty is sum_{u<v} C(popcount(adj[c][u] & adj[c][v]), 2)
        
        def compute_penalty_color(c):
            total = 0
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u + 1, n):
                    common = mask_u & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
            return total
        
        def compute_total_penalty():
            return sum(compute_penalty_color(c) for c in range(num_colors))
        
        current_penalty = compute_total_penalty()
        best_penalty = current_penalty
        best_col = col.copy()
        
        # Precompute edge index lookup: edge_idx[(i,j)] = index into edges array
        edge_idx = {}
        for k, (i, j) in enumerate(edges):
            edge_idx[(i, j)] = k
        
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and time.time() - start_time < time_budget:
            if current_penalty == 0:
                return col.copy(), True
            
            # Pick random edge
            k = random.randint(0, m - 1)
            i, j = edges[k]
            old_c = col[k]
            new_c = random.randint(0, num_colors - 1)
            if new_c == old_c:
                continue
            
            # Compute penalty delta
            # Remove edge (i,j) from old_c:
            #   Pairs affected: (i,v) for v!=j, (j,v) for v!=i, (i,j)
            delta = 0
            
            # Helper: contribution of pair (u,v) in color c
            def pair_contrib(u, v, c):
                common = adj[c][u] & adj[c][v]
                k2 = common.bit_count()
                return k2 * (k2 - 1) // 2 if k2 >= 2 else 0
            
            # Remove from old_c
            # Pair (i, j):
            delta -= pair_contrib(i, j, old_c)
            # Pairs (i, v) for v != i,j:
            for v in range(n):
                if v == i or v == j:
                    continue
                old_contrib = pair_contrib(i, v, old_c)
                # After removal, N_oldc(i) loses j
                # New common of i and v: remove j from N(i), so common loses j if j was in N(v)
                j_in_nv = bool(adj[old_c][v] & (1 << j))
                old_k = (adj[old_c][i] & adj[old_c][v]).bit_count()
                new_k = old_k - (1 if j_in_nv else 0)
                new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
                delta += new_contrib - old_contrib
                
            # Pairs (j, v) for v != i,j:
            for v in range(n):
                if v == i or v == j:
                    continue
                old_contrib = pair_contrib(j, v, old_c)
                i_in_nv = bool(adj[old_c][v] & (1 << i))
                old_k = (adj[old_c][j] & adj[old_c][v]).bit_count()
                new_k = old_k - (1 if i_in_nv else 0)
                new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
                delta += new_contrib - old_contrib
            
            # Add to new_c
            # Pair (i, j):
            delta += pair_contrib(i, j, new_c)
            # Pairs (i, v) for v != i,j:
            for v in range(n):
                if v == i or v == j:
                    continue
                old_contrib = pair_contrib(i, v, new_c)
                j_in_nv = bool(adj[new_c][v] & (1 << j))
                old_k = (adj[new_c][i] & adj[new_c][v]).bit_count()
                new_k = old_k + (1 if j_in_nv else 0)
                new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
                delta += new_contrib - old_contrib
                
            # Pairs (j, v) for v != i,j:
            for v in range(n):
                if v == i or v == j:
                    continue
                old_contrib = pair_contrib(j, v, new_c)
                i_in_nv = bool(adj[new_c][v] & (1 << i))
                old_k = (adj[new_c][j] & adj[new_c][v]).bit_count()
                new_k = old_k + (1 if i_in_nv else 0)
                new_contrib = new_k * (new_k - 1) // 2 if new_k >= 2 else 0
                delta += new_contrib - old_contrib
            
            new_penalty = current_penalty + delta
            
            if new_penalty < current_penalty:
                # Apply the move
                col[k] = new_c
                # Update adjacency
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                current_penalty = new_penalty
                no_improve = 0
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_col = col.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        # Check if best is valid
        # Rebuild adj from best_col
        adj_check = [[0] * n for _ in range(num_colors)]
        for k, (i, j) in enumerate(edges):
            c = best_col[k]
            adj_check[c][i] |= (1 << j)
            adj_check[c][j] |= (1 << i)
        
        is_valid = True
        for c in range(num_colors):
            for u in range(n):
                for v in range(u + 1, n):
                    common = (adj_check[c][u] & adj_check[c][v]).bit_count()
                    if common >= 2:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                break
        
        return best_col, is_valid
    
    # Try n=26 with multiple restarts
    n_target = 26
    
    while time.time() < deadline - 5:
        time_budget = min(deadline - time.time() - 3, 30)
        col, is_valid = try_solve_n(n_target, 5, time_budget)
        if is_valid:
            # Convert to output format
            res_colors = []
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    edges.append((i, j))
                    res_colors.append(int(col[len(edges) - 1]))
            return {'n': n_target, 'colors': res_colors}
    
    # Fallback: try n=25
    n_fallback = 25
    if time.time() < deadline - 10:
        time_budget = deadline - time.time() - 5
        col, is_valid = try_solve_n(n_fallback, 5, time_budget)
        if is_valid:
            res_colors = []
            for i in range(n_fallback):
                for j in range(i + 1, n_fallback):
                    k = i * (2 * n_fallback - i - 1) // 2 + (j - i - 1)
                    res_colors.append(int(col[k]))
            return {'n': n_fallback, 'colors': res_colors}
    
    # Ultimate fallback: n=11 seed solution
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
