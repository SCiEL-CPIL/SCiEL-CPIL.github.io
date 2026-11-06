import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 145.0  # 5s safety margin

    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    def local_search(n, max_time=10.0):
        """
        Fast Best-Improvement Local Search with Bitmasks.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build adjacency bitsets: adj[c][v] is a bitmask of neighbors in color c
        adj = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # Compute common neighbor counts: common[c][i][j]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    cnt = bin(adj[c][i] & adj[c][j]).count('1')
                    common[c][i][j] = cnt
                    common[c][j][i] = cnt

        def calc_cost():
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        k = common[c][i][j]
                        if k >= 2:
                            cost += k * (k - 1) // 2
            return cost

        current_cost = calc_cost()
        best_cost = current_cost
        best_colors = colors[:]
        
        t_end = start + max_time
        while time.time() < t_end:
            improved = False
            best_delta = 0
            best_edge_idx = -1
            best_new_color = -1
            
            # Find the best move
            for idx in range(total_edges):
                i, j = edges[idx]
                old_c = colors[idx]
                
                # Calculate delta for changing to each new color
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    delta = 0
                    # Remove from old_c
                    # For each k in N(j, old_c) \ {i}: common[old_c][i][k] -= 1
                    # Cost change: -(common[old_c][i][k] - 1)
                    mask_j_old = adj[old_c][j] & ~(1 << i)
                    k = 0
                    while mask_j_old:
                        if mask_j_old & 1:
                            old_val = common[old_c][i][k]
                            if old_val > 0:
                                delta -= (old_val - 1)
                            # common[old_c][i][k] -= 1 (temporarily for delta calc, not updating yet)
                        mask_j_old >>= 1
                        k += 1
                        
                    mask_i_old = adj[old_c][i] & ~(1 << j)
                    k = 0
                    while mask_i_old:
                        if mask_i_old & 1:
                            old_val = common[old_c][j][k]
                            if old_val > 0:
                                delta -= (old_val - 1)
                        mask_i_old >>= 1
                        k += 1
                        
                    # Add to new_c
                    # For each k in N(j, new_c) \ {i}: common[new_c][i][k] += 1
                    # Cost change: +common[new_c][i][k]
                    mask_j_new = adj[new_c][j] & ~(1 << i)
                    k = 0
                    while mask_j_new:
                        if mask_j_new & 1:
                            old_val = common[new_c][i][k]
                            delta += old_val
                        mask_j_new >>= 1
                        k += 1
                        
                    mask_i_new = adj[new_c][i] & ~(1 << j)
                    k = 0
                    while mask_i_new:
                        if mask_i_new & 1:
                            old_val = common[new_c][j][k]
                            delta += old_val
                        mask_i_new >>= 1
                        k += 1
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_edge_idx = idx
                        best_new_color = new_c
            
            if best_delta < 0:
                # Apply the best move
                idx = best_edge_idx
                i, j = edges[idx]
                old_c = colors[idx]
                new_c = best_new_color
                
                # Update adj and common
                # Remove from old_c
                mask_j_old = adj[old_c][j] & ~(1 << i)
                k = 0
                while mask_j_old:
                    if mask_j_old & 1:
                        common[old_c][i][k] -= 1
                        common[old_c][k][i] -= 1
                    mask_j_old >>= 1
                    k += 1
                    
                mask_i_old = adj[old_c][i] & ~(1 << j)
                k = 0
                while mask_i_old:
                    if mask_i_old & 1:
                        common[old_c][j][k] -= 1
                        common[old_c][k][j] -= 1
                    mask_i_old >>= 1
                    k += 1
                    
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                
                # Add to new_c
                mask_j_new = adj[new_c][j] & ~(1 << i)
                k = 0
                while mask_j_new:
                    if mask_j_new & 1:
                        common[new_c][i][k] += 1
                        common[new_c][k][i] += 1
                    mask_j_new >>= 1
                    k += 1
                        
                mask_i_new = adj[new_c][i] & ~(1 << j)
                k = 0
                while mask_i_new:
                    if mask_i_new & 1:
                        common[new_c][j][k] += 1
                        common[new_c][k][j] += 1
                    mask_i_new >>= 1
                    k += 1
                    
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
                colors[idx] = new_c
                current_cost += best_delta
                improved = True
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
                if current_cost == 0:
                    return best_colors
            else:
                # No improvement found. Add some randomness to escape local minima.
                # Pick a random edge and change its color randomly.
                idx = random.randint(0, total_edges - 1)
                i, j = edges[idx]
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                # Calculate delta for this random move
                delta = 0
                mask_j_old = adj[old_c][j] & ~(1 << i)
                k = 0
                while mask_j_old:
                    if mask_j_old & 1:
                        old_val = common[old_c][i][k]
                        if old_val > 0:
                            delta -= (old_val - 1)
                    mask_j_old >>= 1
                    k += 1
                    
                mask_i_old = adj[old_c][i] & ~(1 << j)
                k = 0
                while mask_i_old:
                    if mask_i_old & 1:
                        old_val = common[old_c][j][k]
                        if old_val > 0:
                            delta -= (old_val - 1)
                    mask_i_old >>= 1
                    k += 1
                    
                mask_j_new = adj[new_c][j] & ~(1 << i)
                k = 0
                while mask_j_new:
                    if mask_j_new & 1:
                        old_val = common[new_c][i][k]
                        delta += old_val
                    mask_j_new >>= 1
                    k += 1
                        
                mask_i_new = adj[new_c][i] & ~(1 << j)
                k = 0
                while mask_i_new:
                    if mask_i_new & 1:
                        old_val = common[new_c][j][k]
                        delta += old_val
                    mask_i_new >>= 1
                    k += 1

                # Apply random move
                mask_j_old = adj[old_c][j] & ~(1 << i)
                k = 0
                while mask_j_old:
                    if mask_j_old & 1:
                        common[old_c][i][k] -= 1
                        common[old_c][k][i] -= 1
                    mask_j_old >>= 1
                    k += 1
                    
                mask_i_old = adj[old_c][i] & ~(1 << j)
                k = 0
                while mask_i_old:
                    if mask_i_old & 1:
                        common[old_c][j][k] -= 1
                        common[old_c][k][j] -= 1
                    mask_i_old >>= 1
                    k += 1
                    
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                
                mask_j_new = adj[new_c][j] & ~(1 << i)
                k = 0
                while mask_j_new:
                    if mask_j_new & 1:
                        common[new_c][i][k] += 1
                        common[new_c][k][i] += 1
                    mask_j_new >>= 1
                    k += 1
                        
                mask_i_new = adj[new_c][i] & ~(1 << j)
                k = 0
                while mask_i_new:
                    if mask_i_new & 1:
                        common[new_c][j][k] += 1
                        common[new_c][k][j] += 1
                    mask_i_new >>= 1
                    k += 1
                    
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
                colors[idx] = new_c
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]

        if best_cost == 0:
            return best_colors
        return None

    # Try to find valid coloring for increasing n
    best_n = 11
    # Fallback coloring for n=11 (known to be valid)
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # We will try n=26 first, then 25, etc.
    for n_try in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on n
        if n_try == 26:
            restarts = 10
            time_per_restart = 10.0
        elif n_try == 25:
            restarts = 5
            time_per_restart = 10.0
        else:
            restarts = 3
            time_per_restart = 5.0
            
        found = False
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
            res = local_search(n_try, max_time=time_per_restart)
            if res is not None:
                if full_check(n_try, res):
                    best_n = n_try
                    best_colors = res
                    found = True
                    break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
