import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialization: Spectral + Hungarian
    # We want to minimize sum a[i,j] b[p[i], p[j]]
    # Approximation: minimize sum (row_sum_a[i] * col_sum_b[p[i]]) + ... 
    # Or use the "rank" heuristic which is robust.
    # Let's use a combination:
    # 1. Spectral embedding of A and B to get a soft assignment, then Hungarian.
    
    # Simplified robust init:
    # Sort facilities by flow intensity (sum of row i of a)
    # Sort locations by distance intensity (sum of row k of b)
    # Match largest flow to smallest distance (central)
    
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Rank matching: High flow -> Low distance (center)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_p = p.copy()
    best_cost = current_cost

    # Helper: Calculate cost
    def calc_cost(p_arr):
        return np.sum(a * b[np.ix_(p_arr, p_arr)])

    # Helper: Check time
    def time_up():
        return time.time() > deadline

    # 2. Local Search: Variable Neighborhood Descent (VND)
    # Neighborhoods:
    # k=1: 2-swaps (O(n^2) candidates)
    # k=2: 3-swaps (O(n^3) candidates, too many for full scan, use sampling)
    # k=3: 4-swaps (even more, use sampling)
    
    # Precompute for 2-swap delta
    # Delta for swapping i and j:
    # Let u = p[i], v = p[j]
    # Delta = sum_{k != i, j} (a[i,k] - a[j,k]) (b[v, p[k]] - b[u, p[k]]) + a[i,j](b[v,u] - b[u,v])
    # We can vectorize the inner sum for all j for a fixed i?
    # Actually, let's just do a vectorized best-improvement scan for 2-swaps.
    
    def local_search_2swap(p_arr, current_cost):
        improved = True
        while improved and not time_up():
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Vectorized calculation of deltas for all pairs?
            # This is tricky to fully vectorize without O(n^3) memory.
            # Let's iterate i, and for each i, compute deltas for all j > i.
            
            for i in range(n - 1):
                if time_up():
                    return p_arr, current_cost, False
                
                u = p_arr[i]
                # d_k = b[v, p[k]] - b[u, p[k]] for all v? No, v depends on j.
                
                # Let's compute the term (a[i,:] - a[j,:]) dot (b[v,p] - b[u,p])
                # This is effectively:
                # Sum_k a[i,k] b[v, p[k]] - Sum_k a[i,k] b[u, p[k]] - Sum_k a[j,k] b[v, p[k]] + Sum_k a[j,k] b[u, p[k]]
                
                # Let M = b[np.ix_(p_arr, p_arr)]
                # Term 1: a[i, :] dot M[:, v]  -> a[i, :] dot b[:, v][p_arr]
                # Term 2: a[i, :] dot M[:, u]
                # Term 3: a[j, :] dot M[:, v]
                # Term 4: a[j, :] dot M[:, u]
                
                # Precompute row sums of A against columns of M?
                # Let's just loop j for each i. n=150, so 150*150/2 = 11250 iterations.
                # Each iteration is O(n) dot product. 1.6M ops. Fast in numpy.
                
                for j in range(i + 1, n):
                    v = p_arr[j]
                    
                    # Vector d = b[v, p_arr] - b[u, p_arr]
                    d = b[v, p_arr] - b[u, p_arr]
                    diff_a = a[i, :] - a[j, :]
                    
                    dot_val = np.dot(diff_a, d)
                    
                    # Corrections for i and j
                    # d[i] = b[v, u] - b[u, u]
                    # d[j] = b[v, v] - b[u, v]
                    di = b[v, u] - b[u, u]
                    dj = b[v, v] - b[u, v]
                    
                    interaction = a[i, j] * (b[v, u] - b[u, v])
                    
                    delta = dot_val - diff_a[i] * di - diff_a[j] * dj + interaction
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p_arr[best_i], p_arr[best_j] = p_arr[best_j], p_arr[best_i]
                current_cost += best_delta
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p_arr.copy()
        
        return p_arr, current_cost, improved

    # Phase 1: 2-swap local search
    p, current_cost, _ = local_search_2swap(p, current_cost)
    
    # Phase 2: 3-swap exploration (Stochastic VND)
    # If we have time, try random 3-swaps to escape local minima.
    # A 3-swap involves swapping positions of i, j, k in a cycle: p[i]->p[j], p[j]->p[k], p[k]->p[i]
    # Or just 3 random swaps? No, a "3-swap" neighborhood usually means moving 3 items.
    # Let's use "3-interchange": pick 3 indices i, j, k and try all 6 permutations of their locations.
    
    if not time_up():
        improved = True
        while improved and not time_up():
            improved = False
            # Sample a number of 3-interchanges
            num_samples = 500
            best_delta = 0
            best_perm = None
            best_indices = None
            
            for _ in range(num_samples):
                if time_up():
                    break
                # Pick 3 distinct indices
                idxs = np.random.choice(n, 3, replace=False)
                i, j, k = idxs
                
                u, v, w = p[i], p[j], p[k]
                
                # Current cost contribution of these 3 nodes with all others?
                # It's easier to just compute the delta for the 6 permutations.
                # But computing full cost is O(n^2) which is 22500 ops. 500 * 22500 = 11M ops. Acceptable.
                # However, we can do incremental.
                
                # Let's just compute the cost change for the 6 permutations of (u,v,w) at (i,j,k).
                # The rest of the matrix remains fixed.
                # Let R be the set of other indices.
                # Cost change = New_Cost(i,j,k) - Old_Cost(i,j,k)
                
                # Old:
                # a[i,j]b[u,v] + a[j,k]b[v,w] + a[k,i]b[w,u] 
                # + sum_{r in R} (a[i,r]b[u,p[r]] + a[j,r]b[v,p[r]] + a[k,r]b[w,p[r]])
                # + sum_{r in R} (a[r,i]b[p[r],u] + a[r,j]b[p[r],v] + a[r,k]b[p[r],w])
                # + a[i,i]b[u,u] + a[j,j]b[v,v] + a[k,k]b[w,w] (diagonals)
                
                # This is complex to vectorize for arbitrary R.
                # Given n=150, a full cost calculation `calc_cost(p_new)` takes ~10-20us in numpy?
                # `np.sum(a * b[np.ix_(p, p)])` is O(n^2) multiplications and additions.
                # 150*150 = 22500. Very fast.
                # So we can just try 6 permutations and pick the best if it's better than current.
                
                current_sub_cost = 0
                # Actually, just checking if ANY permutation is better requires 6 full cost calcs?
                # No, we can calculate the delta.
                
                # Let's just do a full cost calc for the best candidate among the 6 perms?
                # That's 6 * 22500 = 135k ops per sample. 500 samples = 67M ops.
                # In Python/NumPy, 67M ops might take 0.1-0.5s.
                # We have ~100s left. So we can do this many times.
                
                perms = [(u, v, w), (u, w, v), (v, u, w), (v, w, u), (w, u, v), (w, v, u)]
                best_perm_cost = np.inf
                best_perm_tuple = None
                
                # Optimization: Only calculate full cost if the "local" change looks promising?
                # Hard to know. Let's just calculate.
                
                for perm in perms:
                    p_new = p.copy()
                    p_new[i], p_new[j], p_new[k] = perm
                    cost_new = calc_cost(p_new)
                    if cost_new < best_perm_cost:
                        best_perm_cost = cost_new
                        best_perm_tuple = perm
                
                delta = best_perm_cost - current_cost
                if delta < best_delta:
                    best_delta = delta
                    best_perm = best_perm_tuple
                    best_indices = (i, j, k)
            
            if best_perm is not None and best_delta < 0:
                i, j, k = best_indices
                p[i], p[j], p[k] = best_perm
                current_cost += best_delta
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
    
    # Phase 3: Return to 2-swap polish after 3-swap moves
    if not time_up():
        p, current_cost, _ = local_search_2swap(p, current_cost)
        
    # Final check
    if current_cost < best_cost:
        best_p = p.copy()
        best_cost = current_cost
        
    return best_p.tolist()
