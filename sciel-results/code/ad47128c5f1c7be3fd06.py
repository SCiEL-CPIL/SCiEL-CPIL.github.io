# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # 150s limit, 10s margin -> 140s
    deadline = start_time + 140.0
    
    A = a.astype(np.float64)
    D = b.astype(np.float64)
    
    # 1. Initialization: Random + Rank heuristic
    # Rank heuristic: assign facilities with high row sums to locations with low row sums (if A is flow, high flow to low distance)
    # Actually, usually high flow goes to low distance.
    a_sum = A.sum(axis=1)
    d_sum = D.sum(axis=1)
    
    # Sort facilities by descending sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending sum (closest to center / least connected)
    central_order = np.argsort(d_sum, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    # Evaluate cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
    
    best_cost = calc_cost(p)
    best_p = p.copy()
    
    # 2. Tabu Search with 2-opt moves
    # Tabu list stores (i, j, swap_count)
    # We will maintain a "memory" of recent swaps.
    
    tabu = {}
    tabu_max_age = 50 # Keep last 50 swaps
    max_iter = 50000 # Arbitrary large, time-limited
    iter_count = 0
    current_cost = best_cost
    current_p = best_p.copy()
    
    # Precompute for faster delta?
    # delta = sum_k A[i,k] (D[p[i], p[j]] - D[p[i], p[i]]) + ...
    # Let's just use full cost evaluation if it's fast enough.
    # n=150, full eval is 150x150 sum. ~22500 ops.
    # 50k iters * 22500 = 1.1e9 ops. Python loops are slow.
    # We need a faster delta or vectorization.
    
    # Let's try to implement a faster delta calculation using NumPy vectors.
    # For swap (i, j):
    # delta = sum_k (A[i,k] + A[k,i]) * (D[p[i], p[j]] - D[p[i], p[i]]) 
    #       + sum_k (A[j,k] + A[k,j]) * (D[p[j], p[i]] - D[p[j], p[j]])
    #       - 2 * (A[i,j] + A[j,i]) * (D[p[i], p[j]] - D[p[j], p[i]])  # Correction for double counting?
    
    # Actually, let's derive properly.
    # C = sum_{u,v} A[u,v] D[p[u], p[v]]
    # Terms involving i or j change.
    # Old:
    # T_ii = A[i,i] D[p[i],p[i]]
    # T_jj = A[j,j] D[p[j],p[j]]
    # T_ij = A[i,j] D[p[i],p[j]]
    # T_ji = A[j,i] D[p[j],p[i]]
    # T_ik = sum_{k!=i,j} A[i,k] D[p[i], p[k]]
    # T_ki = sum_{k!=i,j} A[k,i] D[p[k], p[i]]
    # T_jk = sum_{k!=i,j} A[j,k] D[p[j], p[k]]
    # T_kj = sum_{k!=i,j} A[k,j] D[p[k], p[j]]
    
    # New (swap pi, pj):
    # T'_ii = A[i,i] D[p[j],p[j]]
    # T'_jj = A[j,j] D[p[i],p[i]]
    # T'_ij = A[i,j] D[p[j],p[i]]
    # T'_ji = A[j,i] D[p[i],p[j]]
    # T'_ik = sum_{k!=i,j} A[i,k] D[p[j], p[k]]
    # T'_ki = sum_{k!=i,j} A[k,i] D[p[k], p[j]]
    # T'_jk = sum_{k!=i,j} A[j,k] D[p[i], p[k]]
    # T'_kj = sum_{k!=i,j} A[k,j] D[p[k], p[i]]
    
    # Delta = New - Old
    # = (A[i,i](D[p[j],p[j]]-D[p[i],p[i]]) + A[j,j](D[p[i],p[i]]-D[p[j],p[j]]))
    # + (A[i,j](D[p[j],p[i]]-D[p[i],p[j]]) + A[j,i](D[p[i],p[j]]-D[p[j],p[i]]))
    # + sum_{k!=i,j} (A[i,k](D[p[j],p[k]]-D[p[i],p[k]]) + A[k,i](D[p[k],p[j]]-D[p[k],p[i]])
    #                  + A[j,k](D[p[i],p[k]]-D[p[j],p[k]]) + A[k,j](D[p[k],p[i]]-D[p[k],p[j]]))
    
    # This is still O(n) per swap if we don't precompute.
    # Precompute:
    # R_out[i] = A[i, :]  (Row i of A)
    # R_in[i] = A[:, i]   (Col i of A)
    # D_row[i] = D[p[i], :] (Row p[i] of D)
    # D_col[i] = D[:, p[i]] (Col p[i] of D)
    
    # Then:
    # sum_{k} A[i,k] (D[p[j],p[k]] - D[p[i],p[k]]) = A[i,:].dot(D[p[j],:] - D[p[i],:])
    # sum_{k} A[k,i] (D[p[k],p[j]] - D[p[k],p[i]]) = A[:,i].dot(D[:,p[j]] - D[:,p[i]])
    # Similarly for j.
    
    # We need to exclude k=i and k=j.
    # The terms for k=i and k=j in the sums above are included, but we handled them explicitly?
    # Let's check.
    # In the explicit part, we handled (i,i), (j,j), (i,j), (j,i).
    # In the sum part, if we sum over all k, we include k=i and k=j.
    # If we include k=i in sum_{k} A[i,k] (D[p[j],p[k]] - D[p[i],p[k]]):
    #   k=i: A[i,i] (D[p[j],p[i]] - D[p[i],p[i]])
    #   This term is NOT in the explicit part (which had D[p[j],p[j]] for i,i).
    # So we can just sum over ALL k, and NOT have the explicit (i,i), (j,j), (i,j), (j,i) parts?
    # Let's see.
    # New - Old = sum_k [ A[i,k](D[p[j],p[k]] - D[p[i],p[k]]) + A[k,i](D[p[k],p[j]] - D[p[k],p[i]])
    #             + A[j,k](D[p[i],p[k]] - D[p[j],p[k]]) + A[k,j](D[p[k],p[i]] - D[p[k],p[j]]) ]
    # 
    # Let's verify for k=i:
    # Term 1: A[i,i] (D[p[j],p[i]] - D[p[i],p[i]])
    # Term 2: A[i,i] (D[p[i],p[j]] - D[p[i],p[i]])
    # Term 3: A[j,i] (D[p[i],p[i]] - D[p[j],p[i]])
    # Term 4: A[i,j] (D[p[i],p[i]] - D[p[i],p[j]])
    # Sum = A[i,i](D[pj,pi] - Dipi) + A[i,i](Dpi,pj - Dipi) + A[j,i](Dpii - Dpj,pi) + A[i,j](Dpii - Dpi,pj)
    # This seems complicated to match against the explicit derivation.
    
    # Let's stick to the safe O(n) delta with NumPy dot products.
    # Precompute D_rows and D_cols for current p.
    # D_rows[i] = D[p[i], :]
    # D_cols[i] = D[:, p[i]]
    
    # When we swap i, j:
    # D_rows[i] becomes D[p[j], :], D_rows[j] becomes D[p[i], :]
    # D_cols[i] becomes D[:, p[j]], D_cols[j] becomes D[:, p[i]]
    # We must update these arrays after every accepted swap.
    
    A_rows = A  # A is (n,n), A_rows[i] is A[i,:]
    A_cols = A.T # A_cols[i] is A[:,i]
    
    # Initialize D_rows, D_cols
    D_rows = np.empty((n, n), dtype=np.float64)
    D_cols = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        D_rows[i] = D[current_p[i]]
        D_cols[i] = D[:, current_p[i]]
    
    # Tabu list: dictionary mapping (i,j) to iteration count
    # We will use a simple list of (i,j) and check age.
    # Or a set of (min(i,j), max(i,j)) to avoid direction dependency.
    # Tabu for 10-20 iterations is good.
    tabu_size = 20
    tabu_moves = [] # List of (i, j, iter)
    
    for iter_count in range(max_iter):
        if time.time() > deadline - 2.0:
            break
            
        # Find best move
        best_delta = 0
        best_i = -1
        best_j = -1
        best_cost_try = current_cost
        
        # Sample random pairs to save time? Or check all?
        # n=150, all pairs = 11175.
        # Delta calc is 4 dot products of size n=150. 4 * 150 * 150 = 90k ops.
        # 11175 * 90k = 1e9 ops per iteration. Too slow.
        # We must sample.
        
        # Sample 100 random pairs per iteration.
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
                
            i, j = np.random.choice(n, 2, replace=False)
            if i == j: continue
            
            # Check tabu
            is_tabu = False
            key = (min(i,j), max(i,j))
            # Optimization: use a set of tabu keys, but we need to expire them.
            # For speed, just check last 20 moves in list.
            for ti, tj, t_iter in tabu_moves:
                if min(ti,tj) == min(i,j) and max(ti,tj) == max(i,j):
                    if iter_count - t_iter < tabu_size:
                        is_tabu = True
                        break
            
            if is_tabu:
                continue
            
            # Calculate delta
            # Delta = A_rows[i] @ (D[p[j],:] - D[p[i],:]) + A_cols[i] @ (D[:,p[j]] - D[:,p[i]])
            #       + A_rows[j] @ (D[p[i],:] - D[p[j],:]) + A_cols[j] @ (D[:,p[i]] - D[:,p[j]])
            
            diff_D_row_i_j = D[p[j]] - D[p[i]] # D[p[j],:] - D[p[i],:]
            diff_D_col_j_i = D[:, p[j]] - D[:, p[i]] # D[:,p[j]] - D[:,p[i]]
            
            diff_D_row_j_i = D[p[i]] - D[p[j]]
            diff_D_col_i_j = D[:, p[i]] - D[:, p[j]]
            
            delta = np.dot(A_rows[i], diff_D_row_i_j) + \
                    np.dot(A_cols[i], diff_D_col_j_i) + \
                    np.dot(A_rows[j], diff_D_row_j_i) + \
                    np.dot(A_cols[j], diff_D_col_i_j)
            
            # Aspiration: if delta < best_cost - current_cost (i.e. new cost < best), allow
            new_cost = current_cost + delta
            if delta < best_delta or (new_cost < best_cost and delta < 0):
                best_delta = delta
                best_i = i
                best_j = j
                best_cost_try = new_cost
            
        if best_i != -1:
            # Apply move
            current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
            current_cost = best_cost_try
            
            # Update D_rows and D_cols
            # Swap rows
            D_rows[best_i], D_rows[best_j] = D_rows[best_j].copy(), D_rows[best_i].copy()
            D_cols[best_i], D_cols[best_j] = D_cols[best_j].copy(), D_cols[best_i].copy()
            
            # Add to tabu
            tabu_moves.append((best_i, best_j, iter_count))
            if len(tabu_moves) > 50: # Keep small
                tabu_moves.pop(0)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                # Maybe restart tabu list on new best?
                # tabu_moves = []
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
