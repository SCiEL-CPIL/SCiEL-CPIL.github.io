import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    best_n = 11
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_colors = colors_fallback

    def get_cost(colors, n):
        """
        Calculate the total number of C4s across all 5 color classes.
        A C4 in color c is determined by a pair of vertices (u, w) having >= 2 common neighbors in color c.
        Number of C4s = sum_{u < w} binom(common_neighbors(u, w), 2)
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
        
        # Build adjacency matrices for each color
        # We can do this efficiently by iterating through edges
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        total_cost = 0.0
        for m in mats:
            # M = m @ m
            # M[u, w] is number of common neighbors
            M = m @ m
            # We need sum_{u < w} (M[u, w] * (M[u, w] - 1)) / 2
            # Let S = sum_{u < w} M[u, w]^2 - sum_{u < w} M[u, w]
            # Sum_{u < w} M[u, w]^2 = 0.5 * (sum(M**2) - sum(M**2, axis=1).sum())? No.
            # sum_{u, w} M[u, w]^2 = sum(M**2).
            # sum_{u == w} M[u, w]^2 = sum(diag(M)**2) = sum(diag(m@m)) = sum(sum(m, axis=1)**2)
            # Let diag_sq = np.sum(np.diag(M)**2)
            # off_diag_sq_sum = np.sum(M**2) - diag_sq
            # sum_{u < w} M[u, w]^2 = off_diag_sq_sum / 2
            
            # sum_{u, w} M[u, w] = sum(M)
            # sum_{u == w} M[u, w] = sum(diag(M))
            # off_diag_sum = sum(M) - sum(diag(M))
            # sum_{u < w} M[u, w] = off_diag_sum / 2
            
            diag_M = np.diag(M)
            sum_M_sq = np.sum(M**2)
            sum_M = np.sum(M)
            
            sum_diag_sq = np.sum(diag_M**2)
            sum_diag = np.sum(diag_M)
            
            sum_off_diag_sq = (sum_M_sq - sum_diag_sq) / 2.0
            sum_off_diag = (sum_M - sum_diag) / 2.0
            
            cost_c = (sum_off_diag_sq - sum_off_diag) / 2.0
            total_cost += cost_c
            
        return total_cost

    # Try to find solutions for larger n, starting from high to low
    # We prioritize finding a valid solution for a large n.
    # If we fail for 26, we try 25, etc.
    # But we must return the best one found.
    
    # To ensure we don't spend all time on n=26 and fail, we allocate time based on n.
    # n=26 is hard. n=12 is easy.
    # Strategy: Try n=12, 13, ... sequentially? 
    # If we find n=12, we have a score > 1.0.
    # If we have time, try n=13, etc.
    # This guarantees an improvement over the seed (which likely gets stuck at 11 or low 10s).
    
    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this n
        # Larger n gets less time if we are running out of time
        # But we want to maximize n.
        # Let's give each n a fixed number of iterations or a time slice.
        # Time slice: 10 seconds for each n? 
        # If we find a valid one, we update best and continue to next n.
        
        num_edges = n * (n - 1) // 2
        max_iter = 20000 if n < 20 else 50000
        
        # Random initialization
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Precompute edge endpoints for fast access?
        # For speed in local search, we need to update cost efficiently.
        # But recomputing cost from scratch with numpy is fast enough for small n.
        # For n=26, E=325. Building matrices takes some time.
        # Let's try to optimize the local search step.
        
        current_cost = get_cost(colors, n)
        
        # Simple Hill Climbing with Random Restarts
        # If stuck, restart.
        
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            if current_cost == 0:
                best_n = n
                best_colors = colors[:]
                break # Found valid for this n, try next n
                
            # Hill climb
            stuck_count = 0
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                    
                colors[e_idx] = new_c
                new_cost = get_cost(colors, n)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    stuck_count = 0
                else:
                    colors[e_idx] = old_c
                    stuck_count += 1
                    
                if stuck_count > 1000:
                    break # Stuck, restart
                    
            if current_cost == 0:
                best_n = n
                best_colors = colors[:]
                break # Found valid for this n, try next n

    return {'n': best_n, 'colors': best_colors}
