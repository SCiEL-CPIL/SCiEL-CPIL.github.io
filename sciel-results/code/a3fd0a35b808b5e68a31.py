"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Hungarian-based rank structure
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    proxy_cost = np.outer(a_row_sums, b_row_sums)
    
    row_ind, col_ind = linear_sum_assignment(proxy_cost)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Fast delta swap using vectorized operations
    def delta_swap(p, i, j):
        pi = p[i]
        pj = p[j]
        
        # Bulk delta for k not in {i,j}:
        # (A[i,k]-A[j,k])*(B[pj,p[k]]-B[pi,p[k]]) + (A[k,i]-A[k,j])*(B[p[k],pj]-B[p[k],pi])
        dB_row = b[pj, p] - b[pi, p]  # B[pj, p[k]] - B[pi, p[k]]
        dB_col = b[p, pj] - b[p, pi]  # B[p[k], pj] - B[p[k], pi]
        dA_row = a[i, :] - a[j, :]
        dA_col = a[:, i] - a[:, j]
        
        bulk_delta = np.dot(dA_row, dB_row) + np.dot(dA_col, dB_col)
        
        # Boundary corrections for k=i and k=j:
        # For k=i:
        # Old: A[i,i]*B[pi,pi] + A[i,i]*B[pi,pi] + A[i,j]*B[pi,pj] + A[i,j]*B[pj,pi]  (from row_i, col_i, row_j, col_j)
        # Actually, let's compute the change for each of the 4 pairs (i,i), (i,j), (j,i), (j,j)
        
        # (i,i): old = A[i,i]*B[pi,pi], new = A[i,i]*B[pj,pj]
        # (i,j): old = A[i,j]*B[pi,pj], new = A[i,j]*B[pj,pi]
        # (j,i): old = A[j,i]*B[pj,pi], new = A[j,i]*B[pi,pj]
        # (j,j): old = A[j,j]*B[pj,pj], new = A[j,j]*B[pi,pi]
        
        # But wait, these are already included in the bulk calculation? No, because for k=i:
        # dB_row[i] = B[pj, p[i]] - B[pi, p[i]] = B[pj, pi] - B[pi, pi]
        # dB_col[i] = B[p[i], pj] - B[p[i], pi] = B[pi, pj] - B[pi, pi]
        # dA_row[i] = A[i,i] - A[j,i]
        # dA_col[i] = A[i,i] - A[i,j]
        
        # The bulk formula gives for k=i:
        # (A[i,i]-A[j,i])*(B[pj,pi]-B[pi,pi]) + (A[i,i]-A[i,j])*(B[pi,pj]-B[pi,pi])
        
        # But the actual change for k=i involves:
        # Row i, v=i: A[i,i]*(B[pj,pj]-B[pi,pi])
        # Row i, v=j: A[i,j]*(B[pj,pi]-B[pi,pj])
        # Col i, u=i: A[i,i]*(B[pj,pj]-B[pi,pi])
        # Col i, u=j: A[j,i]*(B[pi,pj]-B[pj,pi])
        # Row j, v=i: A[j,i]*(B[pi,pj]-B[pj,pi])
        # Row j, v=j: A[j,j]*(B[pi,pi]-B[pj,pj])
        # Col j, u=i: A[i,j]*(B[pj,pi]-B[pi,pj])
        # Col j, u=j: A[j,j]*(B[pi,pi]-B[pj,pj])
        
        # This is getting complex. Let me just compute old and new for the 4 special pairs directly.
        
        # Actually, the bulk formula already includes the contributions from k=i and k=j in a simplified way.
        # The issue is that the bulk formula assumes p_new[k] = p[k] for all k, but for k=i,j, p_new changes.
        
        # Let me re-derive more carefully.
        
        # Total delta = sum over all (u,v) where u in {i,j} or v in {i,j} of A[u,v]*(B[p_new[u],p_new[v]] - B[p[u],p[v]])
        
        # The set of (u,v) is: row i (all v), col i (all u), row j (all v), col j (all u), minus overlaps.
        
        # Overlaps: (i,i) in row i and col i, (i,j) in row i and col j, (j,i) in row j and col i, (j,j) in row j and col j.
        
        # So delta = 
        #   sum_v A[i,v]*(B[p_new[i],p_new[v]] - B[pi,p[v]])
        # + sum_u A[u,i]*(B[p_new[u],p_new[i]] - B[p[u],pi])
        # + sum_v A[j,v]*(B[p_new[j],p_new[v]] - B[pj,p[v]])
        # + sum_u A[u,j]*(B[p_new[u],p_new[j]] - B[p[u],pj])
        # - A[i,i]*(B[pj,pj]-B[pi,pi])  # (i,i) counted in row i and col i
        # - A[i,j]*(B[pj,pi]-B[pi,pj])  # (i,j) counted in row i (v=j) and col j (u=i)
        # - A[j,i]*(B[pi,pj]-B[pj,pi])  # (j,i) counted in row j (v=i) and col i (u=j)
        # - A[j,j]*(B[pi,pi]-B[pj,pj])  # (j,j) counted in row j and col j
        
        # Now, p_new[i]=pj, p_new[j]=pi, p_new[k]=p[k] for k not in {i,j}.
        
        # Row i term: sum_v A[i,v]*(B[pj,p_new[v]] - B[pi,p[v]])
        # = A[i,i]*(B[pj,pj]-B[pi,pi]) + A[i,j]*(B[pj,pi]-B[pi,pj]) + sum_{k not i,j} A[i,k]*(B[pj,p[k]]-B[pi,p[k]])
        
        # Col i term: sum_u A[u,i]*(B[p_new[u],pj] - B[p[u],pi])
        # = A[i,i]*(B[pj,pj]-B[pi,pi]) + A[j,i]*(B[pi,pj]-B[pj,pi]) + sum_{k not i,j} A[k,i]*(B[p[k],pj]-B[p[k],pi])
        
        # Row j term: sum_v A[j,v]*(B[pi,p_new[v]] - B[pj,p[v]])
        # = A[j,i]*(B[pi,pj]-B[pj,pi]) + A[j,j]*(B[pi,pi]-B[pj,pj]) + sum_{k not i,j} A[j,k]*(B[pi,p[k]]-B[pj,p[k]])
        
        # Col j term: sum_u A[u,j]*(B[p_new[u],pi] - B[p[u],pj])
        # = A[i,j]*(B[pj,pi]-B[pi,pj]) + A[j,j]*(B[pi,pi]-B[pj,pj]) + sum_{k not i,j} A[k,j]*(B[p[k],pi]-B[p[k],pj])
        
        # Sum all four:
        # (i,i) terms: 2*A[i,i]*(B[pj,pj]-B[pi,pi])
        # (i,j) terms: A[i,j]*(B[pj,pi]-B[pi,pj]) + A[i,j]*(B[pj,pi]-B[pi,pj]) = 2*A[i,j]*(B[pj,pi]-B[pi,pj])
        # (j,i) terms: A[j,i]*(B[pi,pj]-B[pj,pi]) + A[j,i]*(B[pi,pj]-B[pj,pi]) = 2*A[j,i]*(B[pi,pj]-B[pj,pi])
        # (j,j) terms: 2*A[j,j]*(B[pi,pi]-B[pj,pj])
        # Bulk (k not i,j):
        #   sum_k A[i,k]*(B[pj,p[k]]-B[pi,p[k]]) + sum_k A[k,i]*(B[p[k],pj]-B[p[k],pi])
        # + sum_k A[j,k]*(B[pi,p[k]]-B[pj,p[k]]) + sum_k A[k,j]*(B[p[k],pi]-B[p[k],pj])
        # = sum_k (A[i,k]-A[j,k])*(B[pj,p[k]]-B[pi,p[k]]) + sum_k (A[k,i]-A[k,j])*(B[p[k],pj]-B[p[k],pi])
        
        # Now subtract the overlaps:
        # - A[i,i]*(B[pj,pj]-B[pi,pi])
        # - A[i,j]*(B[pj,pi]-B[pi,pj])
        # - A[j,i]*(B[pi,pj]-B[pj,pi])
        # - A[j,j]*(B[pi,pi]-B[pj,pj])
        
        # Final delta:
        # (i,i): 2*A[i,i]*(B[pj,pj]-B[pi,pi]) - A[i,i]*(B[pj,pj]-B[pi,pi]) = A[i,i]*(B[pj,pj]-B[pi,pi])
        # (i,j): 2*A[i,j]*(B[pj,pi]-B[pi,pj]) - A[i,j]*(B[pj,pi]-B[pi,pj]) = A[i,j]*(B[pj,pi]-B[pi,pj])
        # (j,i): 2*A[j,i]*(B[pi,pj]-B[pj,pi]) - A[j,i]*(B[pi,pj]-B[pj,pi]) = A[j,i]*(B[pi,pj]-B[pj,pi])
        # (j,j): 2*A[j,j]*(B[pi,pi]-B[pj,pj]) - A[j,j]*(B[pi,pi]-B[pj,pj]) = A[j,j]*(B[pi,pi]-B[pj,pj])
        # Bulk: as above
        
        # So:
        delta_ii = a[i,i] * (b[pj,pj] - b[pi,pi])
        delta_ij = a[i,j] * (b[pj,pi] - b[pi,pj])
        delta_ji = a[j,i] * (b[pi,pj] - b[pj,pi])
        delta_jj = a[j,j] * (b[pi,pi] - b[pj,pj])
        
        bulk = np.dot(dA_row, dB_row) + np.dot(dA_col, dB_col)
        
        # But wait, the bulk includes k=i and k=j. We need to exclude them from bulk and add the correct boundary terms.
        # The bulk formula for k=i:
        # dA_row[i]*dB_row[i] + dA_col[i]*dB_col[i]
        # = (A[i,i]-A[j,i])*(B[pj,pi]-B[pi,pi]) + (A[i,i]-A[i,j])*(B[pi,pj]-B[pi,pi])
        
        # But the actual contribution for k=i should be:
        # From row i, v=i: A[i,i]*(B[pj,pj]-B[pi,pi])
        # From row i, v=j: A[i,j]*(B[pj,pi]-B[pi,pj])
        # From col i, u=i: A[i,i]*(B[pj,pj]-B[pi,pi])
        # From col i, u=j: A[j,i]*(B[pi,pj]-B[pj,pi])
        # From row j, v=i: A[j,i]*(B[pi,pj]-B[pj,pi])
        # From row j, v=j: A[j,j]*(B[pi,pi]-B[pj,pj])
        # From col j, u=i: A[i,j]*(B[pj,pi]-B[pi,pj])
        # From col j, u=j: A[j,j]*(B[pi,pi]-B[pj,pj])
        # Minus overlaps: -A[i,i]*(B[pj,pj]-B[pi,pi]) - A[i,j]*(B[pj,pi]-B[pi,pj]) - A[j,i]*(B[pi,pj]-B[pj,pi]) - A[j,j]*(B[pi,pi]-B[pj,pj])
        
        # This is getting too complex. Let me just use the direct computation for the 4 special pairs and exclude i,j from bulk.
        
        # Create masks to exclude i and j from bulk
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        bulk_delta = np.dot(dA_row[mask], dB_row[mask]) + np.dot(dA_col[mask], dB_col[mask])
        
        boundary = delta_ii + delta_ij + delta_ji + delta_jj
        
        return bulk_delta + boundary

    # 3. Simulated Annealing with adaptive restarts
    start_time = time.time()
    time_limit = 115.0
    
    T = 500.0
    cooling_rate = 0.99995
    best_T_for_restart = 50.0
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        
        delta = delta_swap(p, i, j)
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
        
        # Adaptive restart: if T is low and no recent improvement, restart from best
        if T < best_T_for_restart and random.random() < 0.01:
            p = best_p.copy()
            current_cost = best_cost
            T = 500.0
    
    # 4. Local Search Polish
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = delta_swap(p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
