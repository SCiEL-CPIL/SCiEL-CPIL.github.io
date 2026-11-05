import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 26
    num_colors = 5
    
    # Edge representation: colors[i][j] for i < j
    # Using a 2D array for speed
    colors = [[0]*n for _ in range(n)]
    
    # Bitset representation for fast C4 counting
    # adj[c][u] is an integer bitmask of neighbors of u in color c
    # We need to update this when colors change
    
    def build_adjacency(col_mat):
        adj = [[0]*n for _ in range(num_colors)]
        for i in range(n):
            for j in range(i+1, n):
                c = col_mat[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def count_c4s_fast(adj):
        total_penalty = 0
        for c in range(num_colors):
            for u in range(n):
                mask_u = adj[c][u]
                for v in range(u+1, n):
                    # Common neighbors of u and v in color c
                    # Note: u and v are not neighbors of themselves in the mask
                    common = mask_u & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total_penalty += k * (k - 1) // 2
        return total_penalty

    def random_init():
        col = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                col[i][j] = random.randint(0, num_colors-1)
                col[j][i] = col[i][j]
        return col

    def try_solve():
        col = random_init()
        adj = build_adjacency(col)
        current_penalty = count_c4s_fast(adj)
        best_local_penalty = current_penalty
        best_local_col = [row[:] for row in col]
        
        no_improve_count = 0
        max_no_improve = 5000 # Adjust based on speed
        
        while time.time() < deadline:
            if current_penalty == 0:
                return col
            
            # Pick a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = col[i][j]
            new_c = random.randint(0, num_colors-1)
            if new_c == old_c:
                continue
            
            # Update colors
            col[i][j] = new_c
            col[j][i] = new_c
            
            # Update adjacency bitsets
            # Remove i from adj[old_c][j] and j from adj[old_c][i]
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            
            # Add i to adj[new_c][j] and j to adj[new_c][i]
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            # Recalculate penalty
            new_penalty = count_c4s_fast(adj)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                no_improve_count = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = [row[:] for row in col]
            else:
                # Revert
                col[i][j] = old_c
                col[j][i] = old_c
                
                # Revert adjacency
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        return best_local_col

    # Run multiple attempts
    attempts = 0
    best_valid_n = 0
    best_valid_colors = None
    
    while time.time() < deadline:
        attempts += 1
        sol = try_solve()
        if sol is not None:
            # Verify with fast count
            adj = build_adjacency(sol)
            if count_c4s_fast(adj) == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}
        
        # If we are running out of time, maybe try smaller n?
        # But the task wants to maximize n.
        # The seed returns 11. If we fail 26, we should return 11 to not score 0.
        
        if time.time() > deadline - 10:
            break

    # Fallback to seed solution (n=11)
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
