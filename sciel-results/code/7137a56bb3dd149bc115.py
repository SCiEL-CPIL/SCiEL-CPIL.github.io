"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Precompute
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0

    def compute_cost(p):
        # sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Initial solution: Hungarian algorithm on linear surrogate
    # Surrogate: facility i has flow centrality a.sum(axis=1), location j has distance centrality b.sum(axis=1)
    # We want to assign high-flow facilities to low-distance locations
    # Cost matrix for Hungarian: C[i,j] = -a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # But this is a rank-1 matrix, so Hungarian will give a specific optimal assignment.
    # Better: use a more nuanced surrogate. Let's use the actual QAP structure.
    # For a linear surrogate, we can use: C[i,j] = sum_k a[i,k] * b[j, k] + sum_k a[k, i] * b[k, j]
    # This is the contribution of facility i at location j if all other facilities were optimally placed.
    # Actually, a simpler and effective surrogate: C[i,j] = a.sum(axis=1)[i] * b.sum(axis=1)[j]
    # Since we want to minimize, we negate for Hungarian (which minimizes).
    
    flow_cent = a.sum(axis=1).astype(np.float64)
    dist_cent = b.sum(axis=1).astype(np.float64)
    
    # Build cost matrix for Hungarian: we want high flow_cent matched with low dist_cent
    # So cost[i,j] = flow_cent[i] * dist_cent[j] (we minimize this)
    C = np.outer(flow_cent, dist_cent)
    
    # Hungarian algorithm minimizes the total cost
    row_ind, col_ind = linear_sum_assignment(C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    best_p = p_hung.copy()
    best_cost = compute_cost(best_p)

    # Vectorized swap delta computation
    # Precompute for faster delta calculation
    # For swap of i and j: delta = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    #                          + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
    #                          + (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    
    def compute_all_swap_deltas(p):
        """Compute delta for all pairs (i,j) with i<j in a vectorized manner."""
        # p is the current permutation
        # We need to compute for each pair (i,j):
        # delta[i,j] = sum_{k!=i,j} (a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
        #            + sum_{k!=i,j} (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])
        #            + (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
        
        # Vectorize over i for fixed j, then loop over j
        deltas = np.zeros((n, n), dtype=np.float64)
        
        # Precompute b_perm = b[p, p] for faster access
        b_perm = b[np.ix_(p, p)]
        
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                
                # mask for k != i, j
                # Instead of creating a mask each time, compute full sum and subtract i,j terms
                # term1 = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
                #        = sum_k (a[i,k]-a[j,k])*(b[v, p[k]]-b[u, p[k]])
                # Let's compute the full sum including k=i and k=j, then correct
                
                # Full sum for term1:
                # sum_k (a[i,k]-a[j,k]) * (b[v, p[k]] - b[u, p[k]])
                # = (a[i,:]-a[j,:]) @ (b[v, p] - b[u, p])
                
                diff_a_row = a[i, :] - a[j, :]  # (n,)
                diff_b_row = b[v, p] - b[u, p]  # (n,)
                term1_full = np.dot(diff_a_row, diff_b_row)
                
                # Subtract k=i term: (a[i,i]-a[j,i])*(b[v,p[i]]-b[u,p[i]]) = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
                # Subtract k=j term: (a[i,j]-a[j,j])*(b[v,p[j]]-b[u,p[j]]) = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
                term1_ki = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                term1_kj = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                term1 = term1_full - term1_ki - term1_kj
                
                # term2 = sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
                # = (a[:,i]-a[:,j]) @ (b[p, v] - b[p, u])
                diff_a_col = a[:, i] - a[:, j]  # (n,)
                diff_b_col = b[p, v] - b[p, u]  # (n,)
                term2_full = np.dot(diff_a_col, diff_b_col)
                
                # Subtract k=i term: (a[i,i]-a[i,j])*(b[p[i],v]-b[p[i],u]) = (a[i,i]-a[i,j])*(b[u,v]-b[u,u])
                # Subtract k=j term: (a[j,i]-a[j,j])*(b[p[j],v]-b[p[j],u]) = (a[j,i]-a[j,j])*(b[v,v]-b[v,u])
                term2_ki = (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
                term2_kj = (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
                term2 = term2_full - term2_ki - term2_kj
                
                term3 = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
                
                deltas[i, j] = term1 + term2 + term3
                deltas[j, i] = deltas[i, j]
        
        return deltas

    # Hill climbing with vectorized deltas and random restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            p = best_p.copy()
        else:
            # Perturb best_p with random swaps
            p = best_p.copy()
            num_swaps = 3 + restart // 5
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]

        # Local search using vectorized delta computation
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Compute all swap deltas
            deltas = compute_all_swap_deltas(p)
            
            # Find the most negative delta
            # Only consider i < j
            iu, ju = np.triu_indices(n, k=1)
            valid_deltas = deltas[iu, ju]
            
            if len(valid_deltas) == 0:
                break
                
            min_idx = np.argmin(valid_deltas)
            min_delta = valid_deltas[min_idx]
            
            if min_delta >= 0:
                break
            
            # Perform the swap
            i_idx = iu[min_idx]
            j_idx = ju[min_idx]
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
            improved = True

        # Re-evaluate
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    # Final polish: one more pass with all swaps
    if time.time() < deadline:
        p = best_p.copy()
        for _ in range(3):  # Multiple passes for thoroughness
            if time.time() > deadline:
                break
            deltas = compute_all_swap_deltas(p)
            iu, ju = np.triu_indices(n, k=1)
            valid_deltas = deltas[iu, ju]
            
            if len(valid_deltas) == 0:
                break
                
            min_idx = np.argmin(valid_deltas)
            min_delta = valid_deltas[min_idx]
            
            if min_delta >= 0:
                break
            
            i_idx = iu[min_idx]
            j_idx = ju[min_idx]
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
        
        cur_cost = compute_cost(p)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
