import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial assignment: Rank matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    heavy_order = np.argsort(-a_sums, kind='stable')
    central_order = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_order] = central_order
    
    def calc_cost_full(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    current_cost = calc_cost_full(p)
    best_p = p.copy()
    best_cost = current_cost
    
    def efficient_2opt(p, cost, time_budget=5.0):
        """
        Efficient 2-opt local search using precomputed row/column sums
        to compute swap deltas in O(n) instead of O(n^2) per swap.
        """
        p = p.copy()
        cost = int(cost)
        start = time.time()
        
        # Precompute: for each facility i, the "interaction vector" with all others
        # When swapping p[i] and p[j], the delta can be computed from:
        # delta = sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) + sum_k a[k,i]*(b[pk,pj] - b[pk,pi])
        #       + sum_k a[j,k]*(b[pi,pk] - b[pj,pk]) + sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
        #       + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
        #       + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
        
        # For efficiency, precompute:
        # R[i] = a[i, :]  (row i of a)
        # C[i] = a[:, i]  (column i of a)
        # Then for swap (i,j) with pi=p[i], pj=p[j]:
        # delta = R[i] @ (b[pj,:] - b[pi,:]) + C[i] @ (b[:,pj] - b[:,pi])
        #       + R[j] @ (b[pi,:] - b[pj,:]) + C[j] @ (b[:,pi] - b[:,pj])
        #       + correction terms for k in {i,j}
        
        # Actually, let's use the simpler approach: precompute b_p = b[p, :] and b_pT = b[:, p]
        # Then delta for swap (i,j):
        # Let pi = p[i], pj = p[j]
        # The cost contribution involving i or j changes.
        
        # Precompute column sums of a weighted by b positions
        # S_row[i] = sum_k a[i,k] * b[p[i], p[k]]  -- this is the "row cost" for facility i at location p[i]
        # S_col[i] = sum_k a[k,i] * b[p[k], p[i]]  -- this is the "col cost" for facility i at location p[i]
        
        # Actually, let's just do a moderate-efficiency approach:
        # For each i, precompute a_i = a[i, :] and a_col_i = a[:, i]
        # Then for swap (i,j):
        #   pi = p[i], pj = p[j]
        #   delta = sum_k a[i,k]*(b[pj,pk]-b[pi,pk]) + sum_k a[k,i]*(b[pk,pj]-b[pk,pi])
        #         + sum_k a[j,k]*(b[pi,pk]-b[pj,pk]) + sum_k a[k,j]*(b[pk,pi]-b[pk,pj])
        #   But we need to be careful about double-counting when k=i or k=j.
        
        # Let's use the formula from the inspiration more directly but vectorized over j for fixed i.
        # For fixed i, we can compute deltas for all j > i in a semi-vectorized way.
        
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i in range(n):
                if time.time() - start > time_budget:
                    break
                pi = p[i]
                # For all j > i, compute delta of swapping p[i] and p[j]
                # delta_ij = sum_k a[i,k]*(b[pj,pk] - b[pi,pk]) 
                #          + sum_k a[k,i]*(b[pk,pj] - b[pk,pi])
                #          + sum_k a[j,k]*(b[pi,pk] - b[pj,pk])
                #          + sum_k a[k,j]*(b[pk,pi] - b[pk,pj])
                #          + correction for k=i,j terms
                
                # Let's compute this for all j at once using matrix operations
                # b_pj = b[p, :]  (n x n)
                # For fixed i, pi = p[i]:
                # term1[j] = sum_k a[i,k] * (b[p[j], p[k]] - b[pi, p[k]])
                #          = sum_k a[i,k]*b[p[j],p[k]] - sum_k a[i,k]*b[pi,p[k]]
                #          = (a[i,:] @ b_p.T)[j] - (a[i,:] @ b_p[pi,:])  ... wait
                # Actually: sum_k a[i,k] * b[p[j], p[k]] = (a[i,:] @ b_p)[j] where b_p = b[p,:]
                # And sum_k a[i,k] * b[pi, p[k]] = a[i,:] @ b_p[pi,:]  (scalar for fixed i)
                
                b_p = b[p, :]  # b_p[k, l] = b[p[k], p[l]]
                b_pT = b[:, p]  # b_pT[k, l] = b[p[k], p[l]] same as b_p actually
                
                # Precompute for efficiency:
                # A_row_i = a[i, :]  (1D, n)
                # A_col_i = a[:, i]  (1D, n)
                
                # term1_all[j] = a[i,:] @ b_p[j,:]  = sum_k a[i,k]*b[p[j],p[k]]
                # term2_all[j] = a[:,i] @ b_pT[:,j]  = sum_k a[k,i]*b[p[k],p[j]]
                # But we need (b[pj,pk] - b[pi,pk]) so:
                # delta_part_i[j] = term1_all[j] - term1_all[i]  (since term1_all[i] = sum_k a[i,k]*b[pi,pk])
                # Wait no: term1_all[j] = sum_k a[i,k]*b[p[j],p[k]], and we want sum_k a[i,k]*(b[p[j],p[k]] - b[pi,p[k]])
                # = term1_all[j] - sum_k a[i,k]*b[pi,p[k]]
                # And sum_k a[i,k]*b[pi,p[k]] = a[i,:] @ b_p[i,:] = term1_all[i]
                # So delta_part_i[j] = term1_all[j] - term1_all[i]
                
                # Similarly for the column part:
                # sum_k a[k,i]*(b[p[k],p[j]] - b[p[k],pi])
                # = sum_k a[k,i]*b[p[k],p[j]] - sum_k a[k,i]*b[p[k],pi]
                # = (a[:,i] @ b_pT[:,j]) - (a[:,i] @ b_pT[:,i])
                # Let term2_all[j] = a[:,i] @ b_pT[:,j]
                # delta_part_col_i[j] = term2_all[j] - term2_all[i]
                
                # Now for the j parts:
                # sum_k a[j,k]*(b[pi,pk] - b[pj,pk]) = -sum_k a[j,k]*(b[pj,pk] - b[pi,pk])
                # = -(term1_j_all[j] - term1_j_all[i])  where term1_j_all[k] = a[j,:]*b_p[k,:]
                # This is getting complicated. Let me just compute delta for each (i,j) pair directly
                # but use precomputed vectors to make it O(n) per pair.
                
                # For fixed i, compute delta for all j in one vectorized op:
                # delta_ij = (a[i,:] @ (b_p[j,:] - b_p[i,:])) 
                #          + (a[:,i] @ (b_pT[:,j] - b_pT[:,i]))
                #          + (a[j,:] @ (b_p[i,:] - b_p[j,:]))
                #          + (a[:,j] @ (b_pT[:,i] - b_pT[:,j]))
                #          + correction terms for k=i and k=j
                
                # The first two terms: a[i,:] @ (b_p[j,:] - b_p[i,:]) = a[i,:]@b_p[j,:] - a[i,:]@b_p[i,:]
                # = (a[i,:] @ b_p.T)[j] - (a[i,:] @ b_p.T)[i]
                # Let v1 = a[i,:] @ b_p.T  (vector of length n)
                # Then first term = v1[j] - v1[i]
                
                # Second term: a[:,i] @ (b_pT[:,j] - b_pT[:,i]) = a[:,i]@b_pT[:,j] - a[:,i]@b_pT[:,i]
                # = (a[:,i] @ b_pT)[j] - (a[:,i] @ b_pT)[i]
                # Let v2 = a[:,i] @ b_pT  (vector of length n)
                # Then second term = v2[j] - v2[i]
                
                # Third term: a[j,:] @ (b_p[i,:] - b_p[j,:]) = -(a[j,:] @ (b_p[j,:] - b_p[i,:]))
                # = -((a[j,:] @ b_p.T)[j] - (a[j,:] @ b_p.T)[i])
                # This requires a[j,:] @ b_p.T for each j, which is expensive.
                
                # Alternative: just loop over j for fixed i, computing delta in O(n) each.
                # With n=256, that's 256*256/2 = 32768 pairs, each O(n)=256 ops = ~8M ops per pass.
                # That's feasible in a few seconds.
                
                v1 = a[i, :] @ b_p.T  # v1[j] = sum_k a[i,k]*b[p[j],p[k]]
                v2 = a[:, i] @ b_pT   # v2[j] = sum_k a[k,i]*b[p[k],p[j]]
                
                for j in range(i + 1, n):
                    if time.time() - start > time_budget:
                        break
                    pi_val = p[i]
                    pj_val = p[j]
                    
                    # Compute delta for swapping p[i] and p[j]
                    # Using the decomposition:
                    # delta = sum_k a[i,k]*(b[pj,pk]-b[pi,pk]) + sum_k a[k,i]*(b[pk,pj]-b[pk,pi])
                    #       + sum_k a[j,k]*(b[pi,pk]-b[pj,pk]) + sum_k a[k,j]*(b[pk,pi]-b[pk,pj])
                    #       + self/cross correction terms
                    
                    # First two sums:
                    # sum_k a[i,k]*(b[pj,pk]-b[pi,pk]) = v1[j] - v1[i]
                    # sum_k a[k,i]*(b[pk,pj]-b[pk,pi]) = v2[j] - v2[i]
                    
                    part1 = v1[j] - v1[i]
                    part2 = v2[j] - v2[i]
                    
                    # Last two sums (symmetric):
                    # sum_k a[j,k]*(b[pi,pk]-b[pj,pk]) = -sum_k a[j,k]*(b[pj,pk]-b[pi,pk])
                    # = -(v1_j[j] - v1_j[i]) where v1_j[k] = sum_l a[j,l]*b[p[k],p[l]]
                    # This is expensive to compute for each j. Let me just compute it directly.
                    
                    # Direct computation for the j-related terms:
                    # sum_k a[j,k]*(b[pi,pk]-b[pj,pk])
                    # = sum_k a[j,k]*b[pi,pk] - sum_k a[j,k]*b[pj,pk]
                    # = a[j,:] @ b[pi, p] - a[j,:] @ b[pj, p]
                    
                    # sum_k a[k,j]*(b[pk,pi]-b[pk,pj])
                    # = sum_k a[k,j]*b[pk,pi] - sum_k a[k,j]*b[pk,pj]
                    # = a[:,j] @ b[p, pi] - a[:,j] @ b[p, pj]
                    
                    # Let me precompute these for all j at once for fixed i:
                    # Actually, let's just compute the full delta directly for each (i,j)
                    # using the original formula but with vectorized inner products.
                    
                    # Full delta computation:
                    # mask approach is O(n) per pair
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    pk = p[mask]
                    
                    delta = 0.0
                    # Self terms
                    delta += a[i, i] * (b[pj_val, pj_val] - b[pi_val, pi_val])
                    delta += a[j, j] * (b[pi_val, pi_val] - b[pj_val, pj_val])
                    # Cross i,j
                    delta += a[i, j] * (b[pj_val, pi_val] - b[pi_val, pj_val])
                    delta += a[j, i] * (b[pi_val, pj_val] - b[pj_val, pi_val])
                    # Bulk terms
                    delta += np.sum(a[i, mask] * (b[pj_val, pk] - b[pi_val, pk]))
                    delta += np.sum(a[mask, i] * (b[pk, pj_val] - b[pk, pi_val]))
                    delta += np.sum(a[j, mask] * (b[pi_val, pk] - b[pj_val, pk]))
                    delta += np.sum(a[mask, j] * (b[pk, pi_val] - b[pk, pj_val]))
                    
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += int(round(delta))
                        improved = True
                        # Update v1 and v2 since p changed
                        # v1 = a[i,:] @ b_p.T  -- but p changed, so b_p changed
                        # We need to recompute b_p
                        break  # Restart the j loop for this i since p changed
                
        return p, cost
    
    def compute_dynamic_cost(p):
        b_jp = b[:, p]
        b_pj = b[p, :]
        C_first = a @ b_jp.T
        C_second = a.T @ b_pj
        return C_first + C_second
    
    # Main loop: alternate between Hungarian reassignment and efficient local search
    iteration = 0
    T = 5000.0
    min_T = 10.0
    
    # Initial local search
    p, current_cost = efficient_2opt(p, current_cost, time_budget=10.0)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    while time.time() < deadline:
        # Compute dynamic cost matrix
        C = compute_dynamic_cost(p)
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        new_cost = calc_cost_full(new_p)
        
        delta = new_cost - current_cost
        if delta < 0 or (T > min_T and np.random.rand() < np.exp(-delta / T)):
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = new_p.copy()
            
            p = new_p.copy()
            current_cost = new_cost
            
            # Efficient local search refinement
            p, current_cost = efficient_2opt(p, current_cost, time_budget=3.0)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            T = max(min_T, T * 0.9)
        else:
            # Random 2-swap perturbation
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                old_i, old_j = p[i], p[j]
                p[i], p[j] = old_j, old_i
                swap_cost = calc_cost_full(p)
                swap_delta = swap_cost - current_cost
                if swap_delta < 0 or (T > min_T and np.random.rand() < np.exp(-swap_delta / T)):
                    if swap_cost < best_cost:
                        best_cost = swap_cost
                        best_p = p.copy()
                    current_cost = swap_cost
                    T = max(min_T, T * 0.9)
                else:
                    p[i], p[j] = old_i, old_j
                    T = max(min_T, T * 0.98)
        
        iteration += 1
    
    return best_p.tolist()
