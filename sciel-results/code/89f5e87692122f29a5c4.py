"""QAP solver for tai256c: rank-matching init + 2-opt local search with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # Use vectorized computation for speed
    # p is a permutation array
    # b_perm[i, j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _greedy_initial(a, b):
    """Generate a good initial solution using greedy construction."""
    n = len(a)
    # Use the rank-matching heuristic as a starting point
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, max_time=None, start_time=None):
    """
    2-opt local search for QAP.
    Tries swapping pairs of assignments and accepts if cost decreases.
    Uses incremental cost updates for efficiency.
    """
    n = len(a)
    if max_time is None:
        max_time = 140.0
    if start_time is None:
        start_time = time.time()
    
    # Precompute row sums of a for faster cost delta computation
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # If we swap p[i] and p[j], the change involves terms where i or j are involved.
    
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    
    # Precompute some useful arrays
    # For each position i, the contribution to cost from row i is:
    # sum_j a[i,j] * b[p[i], p[j]]
    # And from column i (as the second index in b):
    # sum_j a[j,i] * b[p[j], p[i]]
    
    # Let's precompute the "row contribution" and "column contribution"
    # Actually, let's just compute the full cost matrix contribution
    
    # For 2-opt, swapping positions i and j:
    # Old terms involving i or j as first index:
    #   sum_k a[i,k] * b[p[i], p[k]]  and  sum_k a[j,k] * b[p[j], p[k]]
    # New terms:
    #   sum_k a[i,k] * b[p[j], p[k]]  and  sum_k a[j,k] * b[p[i], p[k]]
    # Similarly for second index (column) contributions.
    
    # Let's precompute for each position i, the vector of a[i,:] and a[:,i]
    # and for each location l, the row b[l,:] and column b[:,l]
    
    # Precompute a_rows and a_cols
    a_rows = a  # a_rows[i] = a[i, :]
    a_cols = a.T  # a_cols[i] = a[:, i]
    
    b_rows = b  # b_rows[l] = b[l, :]
    b_cols = b.T  # b_cols[l] = b[:, l]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50  # Stop if no improvement for this many full passes
    
    while improved and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        if time.time() - start_time > max_time:
            break
        
        # Compute current contribution vectors
        # For each position i, compute:
        # row_contrib[i] = sum_k a[i,k] * b[p[i], p[k]]
        # col_contrib[i] = sum_k a[k,i] * b[p[k], p[i]]
        
        # row_contrib[i] = a[i,:] @ b[p[i], p[:]]
        # col_contrib[i] = a[:,i] @ b[p[:], p[i]]
        
        p_arr = current_p
        b_at_p = b[np.ix_(p_arr, p_arr)]
        
        # row_contrib[i] = sum_k a[i,k] * b_at_p[i,k]
        row_contrib = np.einsum('ik,ik->i', a, b_at_p)
        # col_contrib[i] = sum_k a[k,i] * b_at_p[k,i]
        col_contrib = np.einsum('ki,ki->i', a, b_at_p)
        
        # Total cost = sum_i (row_contrib[i] + col_contrib[i]) / 2? No...
        # Actually cost = sum_{i,j} a[i,j] * b[p[i],p[j]]
        # = sum_i sum_j a[i,j] * b[p[i],p[j]]
        # row_contrib[i] = sum_j a[i,j] * b[p[i],p[j]]
        # col_contrib[i] = sum_j a[j,i] * b[p[j],p[i]]
        # So total = sum_i row_contrib[i] = sum_i col_contrib[i]
        # Let's verify: sum_i row_contrib[i] = sum_i sum_j a[i,j]*b[p[i],p[j]] = cost. Correct.
        
        # For swapping positions i and j:
        # New row_contrib for i: sum_k a[i,k] * b[p[j], p[k]]  (for k != i,j)
        #                       + a[i,i]*b[p[j],p[i]] + a[i,j]*b[p[j],p[j]]
        # Old row_contrib for i: sum_k a[i,k] * b[p[i], p[k]]
        #                       = a[i,i]*b[p[i],p[i]] + a[i,j]*b[p[i],p[j]] + sum_{k!=i,j} a[i,k]*b[p[i],p[k]]
        
        # This is getting complex. Let's use a simpler approach:
        # Just compute the cost delta directly for each pair.
        
        # For efficiency, let's precompute the full cost matrix C[i,j] = a[i,:] @ b_rows[p[i]] 
        # Actually, let's just do pairwise swaps with direct cost computation.
        
        # For n=256, there are 256*255/2 = 32640 pairs per pass.
        # Each swap evaluation can be done in O(n) time.
        # 32640 * 256 = ~8.3M operations per pass, which should be fast enough.
        
        for i in range(n):
            if time.time() - start_time > max_time:
                return current_p, current_cost
            
            for j in range(i + 1, n):
                # Try swapping positions i and j
                # Compute cost delta
                # Old: terms involving i or j
                # New: after swap
                
                pi = p_arr[i]
                pj = p_arr[j]
                
                # The cost change from swapping p[i] and p[j]:
                # We need to recompute all terms where i or j appears as first or second index.
                
                # Terms where i is first index:
                # Old: sum_k a[i,k] * b[pi, p[k]]
                # New: sum_k a[i,k] * b[pj, p[k]]
                # Delta_i_row = sum_k a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
                
                # Terms where j is first index:
                # Old: sum_k a[j,k] * b[pj, p[k]]
                # New: sum_k a[j,k] * b[pi, p[k]]
                # Delta_j_row = sum_k a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
                
                # Terms where i is second index:
                # Old: sum_k a[k,i] * b[p[k], pi]
                # New: sum_k a[k,i] * b[p[k], pj]
                # Delta_i_col = sum_k a[k,i] * (b[p[k], pj] - b[p[k], pi])
                
                # Terms where j is second index:
                # Old: sum_k a[k,j] * b[p[k], pj]
                # New: sum_k a[k,j] * b[p[k], pi]
                # Delta_j_col = sum_k a[k,j] * (b[p[k], pi] - b[p[k], pj])
                
                # Total delta = Delta_i_row + Delta_j_row + Delta_i_col + Delta_j_col
                # But we double count the (i,j) and (j,i) terms? Let's check.
                # When k=j in Delta_i_row: a[i,j] * (b[pj, pj] - b[pi, pj])
                # When k=i in Delta_j_row: a[j,i] * (b[pi, pi] - b[pj, pi])
                # When k=j in Delta_i_col: a[j,i] * (b[pj, pj] - b[pj, pi])
                # When k=i in Delta_j_col: a[i,j] * (b[pi, pi] - b[pi, pj])
                # These are all distinct terms, so no double counting issue.
                
                # Let's compute these using vectorized operations
                p_k = p_arr  # current permutation
                
                # Delta_i_row = a[i,:] @ (b[pj,:] - b[pi,:]) [but only where p[k] varies]
                # Actually: sum_k a[i,k] * b[pj, p[k]] - sum_k a[i,k] * b[pi, p[k]]
                # = a[i,:] @ b[pj, p_k] - a[i,:] @ b[pi, p_k]
                
                delta_i_row = np.dot(a[i, :], b[pj, p_k]) - np.dot(a[i, :], b[pi, p_k])
                delta_j_row = np.dot(a[j, :], b[pi, p_k]) - np.dot(a[j, :], b[pj, p_k])
                delta_i_col = np.dot(a[:, i], b[p_k, pj]) - np.dot(a[:, i], b[p_k, pi])
                delta_j_col = np.dot(a[:, j], b[p_k, pi]) - np.dot(a[:, j], b[p_k, pj])
                
                total_delta = delta_i_row + delta_j_row + delta_i_col + delta_j_col
                
                if total_delta < 0:
                    # Accept the swap
                    p_arr[i], p_arr[j] = p_arr[j], p_arr[i]
                    current_cost += total_delta
                    improved = True
                    no_improve_count = 0
                    break  # Restart the j loop with updated permutation
        
        if not improved:
            no_improve_count += 1
    
    return p_arr, current_cost


def solve(a, b):
    """Solve the QAP instance using greedy init + 2-opt local search."""
    start_time = time.time()
    n = len(a)
    
    # Generate initial solution
    p = _greedy_initial(a, b)
    
    # Run local search with time budget
    max_time = 140.0  # Leave 10s safety margin
    p, cost = _local_search_2opt(a, b, p, max_time=max_time, start_time=start_time)
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > max_time - 5:
        return p.tolist()
    
    return p.tolist()
# EVOLVE-BLOCK-END
