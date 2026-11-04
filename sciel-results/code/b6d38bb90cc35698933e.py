"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialization: Rank matching (greedy heuristic)
    # Sort facilities by total flow (importance) and locations by total distance (centrality)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Heuristic: Match high flow facilities to low distance (central) locations
    # Note: In QAP, we want high flow * low distance.
    # So sort a descending, b ascending.
    heavy_order = np.argsort(-a_sums, kind="stable")
    central_order = np.argsort(b_sums, kind="stable")
    
    p = np.zeros(n, dtype=np.int64)
    for i, loc in zip(heavy_order, central_order):
        p[i] = loc
        
    # Ensure p is a valid permutation (it is by construction)
    
    # 2. Precompute for fast cost calculation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    def calc_cost(p):
        # Vectorized calculation
        # b_perm[i, j] = b[p[i], p[j]]
        # This creates a dense n x n matrix, which is 150x150 = 22500 elements.
        # Summing element-wise product with a.
        # For n=150, this is fast enough for initialization and occasional checks.
        # But for SA loop, we need incremental updates.
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 3. Simulated Annealing
    # Move: 3-cycle (i, j, k) -> p[i], p[j], p[k] become p[k], p[i], p[j]
    # Or swap (i, j). 
    # 3-cycles generate the full symmetric group and can escape 2-opt local minima.
    # However, calculating delta for 3-cycles is O(n) per move.
    # Swaps are O(n) per move too.
    # Let's stick to swaps for speed, but use SA's stochastic acceptance to escape.
    # Actually, the directive asks for "large-step" moves. 
    # Let's implement 3-cycles as the primary move to satisfy "large-step" and structural change.
    
    # Precompute row sums of B? No, B is permuted.
    # Let's precompute the current assignment matrix Bp = b[np.ix_(p, p)]? 
    # That's expensive to update.
    
    # Better approach for incremental cost update in SA:
    # Maintain p.
    # To evaluate a move, we need the change in cost.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    
    # For a swap (u, v) where p[u]=x, p[v]=y:
    # Delta = sum_{k != u, v} a[k, u] (b[p[k], y] - b[p[k], x]) + a[u, k] (b[y, p[k]] - b[x, p[k]])
    #        + a[u, v] (b[y, y] - b[x, x]) + a[v, u] (b[x, x] - b[y, y]) 
    #        + a[u, u] (b[x, x] - b[y, x]) ... wait, this is getting complex.
        
    # Let's use a simpler O(n) update for swaps.
    # Let x = p[u], y = p[v].
    # Terms involving u or v in the sum:
    # 1. i=u, j=v: a[u,v] * b[x,y] -> a[u,v] * b[y,x]
    # 2. i=v, j=u: a[v,u] * b[y,x] -> a[v,u] * b[x,y]
    # 3. i=u, j=k (k!=v): a[u,k] * b[x, p[k]] -> a[u,k] * b[y, p[k]]
    # 4. i=k, j=u (k!=v): a[k,u] * b[p[k], x] -> a[k,u] * b[p[k], y]
    # 5. i=v, j=k (k!=u): a[v,k] * b[y, p[k]] -> a[v,k] * b[x, p[k]]
    # 6. i=k, j=v (k!=u): a[k,v] * b[p[k], y] -> a[k,v] * b[p[k], x]
    # 7. i=u, j=u: a[u,u] * b[x,x] -> a[u,u] * b[y,y]  (if we allow i=j, which we do in QAP sum)
    # 8. i=v, j=v: a[v,v] * b[y,y] -> a[v,v] * b[x,x]
    
    # Wait, the standard QAP definition sum_{i,j} includes i=j.
    # Let's derive the delta for swapping positions u and v in permutation p.
    # Let x = p[u], y = p[v].
    # Old terms:
    # T_old = sum_{k} a[u,k] b[x, p[k]] + sum_{k} a[k,u] b[p[k], x] 
    #       + sum_{k} a[v,k] b[y, p[k]] + sum_{k} a[k,v] b[p[k], y]
    #       - a[u,u]b[x,x] - a[v,v]b[y,y] - a[u,v]b[x,y] - a[v,u]b[y,x] 
    #       (subtracted because they were counted twice in the sums above if k=u or k=v)
    #       Actually, it's easier to just sum the changes for all pairs (i,j) where i or j is u or v.
    
    # Let's define the contribution of a pair (i,j) as C(i,j) = a[i,j] * b[p[i], p[j]].
    # Swap p[u] and p[v]. Let x=p[u], y=p[v].
    # Changed pairs:
    # (u, u): a[u,u]*(b[x,x] - b[y,y])
    # (v, v): a[v,v]*(b[y,y] - b[x,x])
    # (u, v): a[u,v]*(b[x,y] - b[y,x])
    # (v, u): a[v,u]*(b[y,x] - b[x,y])
    # (u, k) for k not u,v: a[u,k]*(b[x, p[k]] - b[y, p[k]])
    # (k, u) for k not u,v: a[k,u]*(b[p[k], x] - b[p[k], y])
    # (v, k) for k not u,v: a[v,k]*(b[y, p[k]] - b[x, p[k]])
    # (k, v) for k not u,v: a[k,v]*(b[p[k], y] - b[p[k], x])
    
    # We can vectorize this.
    
    def swap_delta(u, v, p):
        x = p[u]
        y = p[v]
        
        # Vectorized calculation of the O(n) terms
        # mask for k not in {u, v}
        # This is tricky to vectorize perfectly without creating large temp arrays, 
        # but n=150 is small.
        
        # Let's just compute the sums directly using numpy dot products on slices.
        # This is O(n) but with numpy overhead.
        
        # Terms involving u (row u)
        # sum_{k!=u,v} a[u,k] * (b[x, p[k]] - b[y, p[k]])
        # = a[u, :] dot (b[x, p] - b[y, p]) - a[u,u]*(b[x,x]-b[y,y]) - a[u,v]*(b[x,y]-b[y,x])
        
        # Let's precompute Bp = b[np.ix_(p, p)]? No, that's O(n^2) to build.
        # But we can access b[x, p] and b[y, p] as vectors.
        
        # diff_u = b[x, p] - b[y, p]  # Vector of length n
        # term_u = np.dot(a[u, :], diff_u)
        
        # diff_v = b[y, p] - b[x, p]  # Vector of length n
        # term_v = np.dot(a[v, :], diff_v)
        
        # Similarly for columns (transposed terms)
        # diff_u_col = b[p, x] - b[p, y]
        # term_u_col = np.dot(a[:, u], diff_u_col)
        
        # diff_v_col = b[p, y] - b[p, x]
        # term_v_col = np.dot(a[:, v], diff_v_col)
        
        # Total delta = term_u + term_v + term_u_col + term_v_col
        # But we have double counted the (u,v) and (v,u) interactions?
        # Let's check.
        # term_u includes k=v: a[u,v]*(b[x,y]-b[y,x])
        # term_u_col includes k=v: a[v,u]*(b[p[v],x]-b[p[v],y]) = a[v,u]*(b[y,x]-b[x,y])
        # term_v includes k=u: a[v,u]*(b[y,x]-b[x,y])
        # term_v_col includes k=u: a[u,v]*(b[p[u],y]-b[p[u],x]) = a[u,v]*(b[x,y]-b[x,x])? No.
        # b[p[u], y] = b[x, y]. b[p[u], x] = b[x, x].
        # So term_v_col includes a[u,v]*(b[x,y] - b[x,x]).
        
        # This is getting error-prone. Let's use a robust O(n^2) check for every K steps?
        # No, n=150, O(n^2) is 22500 ops. 10000 steps is 225M ops. Might be too slow in Python.
        
        # Let's stick to the vectorized O(n) swap delta. It's a standard technique.
        
        # Vector b_x = b[x, :]
        # Vector b_y = b[y, :]
        # Vector p_vec = p
        
        # Row u contribution:
        # sum_k a[u,k] * (b[x, p[k]] - b[y, p[k]])
        # = a[u, :] @ (b[x, p] - b[y, p])
        
        # Row v contribution:
        # sum_k a[v,k] * (b[y, p[k]] - b[x, p[k]])
        # = a[v, :] @ (b[y, p] - b[x, p])
        
        # Col u contribution:
        # sum_k a[k,u] * (b[p[k], x] - b[p[k], y])
        # = a[:, u] @ (b[p, x] - b[p, y])
        
        # Col v contribution:
        # sum_k a[k,v] * (b[p[k], y] - b[p[k], x])
        # = a[:, v] @ (b[p, y] - b[p, x])
        
        # The sum of these 4 terms accounts for all changes in the matrix product A * Bp.
        # Note: The diagonal terms (u,u) and (v,v) are included in these sums.
        # The cross terms (u,v) and (v,u) are included in these sums.
        # Are they double counted?
        # The cost is sum_{i,j} a[i,j] b[p[i], p[j]].
        # The change is sum_{i,j} a[i,j] (b[p'_i], p'_j] - b[p[i], p[j]])
        # Only terms where i or j is u or v change.
        # The 4 dot products cover:
        # 1. i=u, all j
        # 2. i=v, all j
        # 3. all i, j=u
        # 4. all i, j=v
        # The intersection (u,u), (u,v), (v,u), (v,v) are counted in multiple terms?
        # (u,u) is in 1 and 3.
        # (u,v) is in 1 and 4.
        # (v,u) is in 2 and 3.
        # (v,v) is in 2 and 4.
        # So yes, the 4 corner terms are double counted.
        # We must subtract them once.
        
        # Corner (u,u): a[u,u] * (b[x,x] - b[y,y])
        # Corner (v,v): a[v,v] * (b[y,y] - b[x,x])
        # Corner (u,v): a[u,v] * (b[x,y] - b[y,x])
        # Corner (v,u): a[v,u] * (b[y,x] - b[x,y])
        
        # So Delta = (Row_u + Row_v + Col_u + Col_v) - (Corner_uu + Corner_vv + Corner_uv + Corner_vu)
        
        b_x = b[x, :]
        b_y = b[y, :]
        
        # b[p, x] is a vector where element k is b[p[k], x]
        # This is b[:, x][p]
        
        row_u_diff = b_x[p] - b_y[p]
        row_v_diff = b_y[p] - b_x[p]
        
        col_u_diff = b[:, x][p] - b[:, y][p]
        col_v_diff = b[:, y][p] - b[:, x][p]
        
        sum_rows = np.dot(a[u, :], row_u_diff) + np.dot(a[v, :], row_v_diff)
        sum_cols = np.dot(a[:, u], col_u_diff) + np.dot(a[:, v], col_v_diff)
        
        total = sum_rows + sum_cols
        
        # Subtract double-counted corners
        corner_uu = a[u, u] * (b[x, x] - b[y, y])
        corner_vv = a[v, v] * (b[y, y] - b[x, x])
        corner_uv = a[u, v] * (b[x, y] - b[y, x])
        corner_vu = a[v, u] * (b[y, x] - b[x, y])
        
        delta = total - (corner_uu + corner_vv + corner_uv + corner_vu)
        
        return int(delta)

    # SA Parameters
    T_start = 1000.0  # Initial temperature
    T_end = 1.0
    cooling_rate = 0.999
    
    T = T_start
    current_p = p.copy()
    current_cost = calc_cost(current_p)
    
    # To speed up, we can precompute b[:, x] lookups? 
    # b is 150x150. Accessing b[x, p] is fast.
    
    # Random seed for reproducibility? No, let it be random.
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate a random swap
        u = rng.integers(n)
        v = rng.integers(n)
        if u == v:
            continue
            
        delta = swap_delta(u, v, current_p)
        new_cost = current_cost + delta
        
        # Metropolis acceptance
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Accept
            current_p[u], current_p[v] = current_p[v], current_p[u]
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        # Optional: Random restarts if stuck?
        # The SA should handle it.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
