"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def solve(a, b):
    n = len(a)
    
    # Convert to float for numerical stability
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    start_time = time.time()
    time_budget = 108.0  # Leave 12 seconds for final local search
    
    # Warm start: match high-flow facilities to central locations
    a_row_sums = A.sum(axis=1)
    b_row_sums = B.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    current_cost = compute_cost(p)
    
    # Efficient swap delta computation using precomputed matrices
    # For QAP cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # When we swap p[i] and p[j], the delta can be computed in O(n)
    
    def get_swap_delta(i, j, p, a, b):
        pi, pj = p[i], p[j]
        if pi == pj:
            return 0
        
        # Terms involving facility i or j with all other facilities
        # Original contribution of i: sum_k a[i,k] * b[pi, p[k]]
        # New contribution of i: sum_k a[i,k] * b[pj, p[k]]
        # Original contribution of j: sum_k a[j,k] * b[pj, p[k]]
        # New contribution of j: sum_k a[j,k] * b[pi, p[k]]
        
        # For k != i, j:
        # delta_k = a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
        #          + a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
        
        # Vectorized:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        k_idx = np.where(mask)[0]
        
        if len(k_idx) > 0:
            pk = p[k_idx]
            delta = 0
            delta += np.sum(a[i, k_idx] * (b[pj, pk] - b[pi, pk]))
            delta += np.sum(a[k_idx, i] * (b[pk, pj] - b[pk, pi]))
            delta += np.sum(a[j, k_idx] * (b[pi, pk] - b[pj, pk]))
            delta += np.sum(a[k_idx, j] * (b[pk, pi] - b[pk, pj]))
        else:
            delta = 0
        
        # Direct i-j interaction terms
        # Original: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
        # New: a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
        delta += a[i, j] * (b[pj, pi] - b[pi, pj])
        delta += a[j, i] * (b[pi, pj] - b[pj, pi])
        
        return delta
    
    # Local search with efficient swap evaluation
    # Use a priority queue or just iterate through pairs
    # For n=256, there are ~32640 pairs per pass
    
    improved = True
    max_passes = 20
    pass_count = 0
    
    while improved and pass_count < max_passes:
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        pass_count += 1
        
        # For each pair (i, j) with i < j, compute swap delta
        # If delta < 0, perform the swap
        for i in range(n):
            if time.time() - start_time > time_budget:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_budget:
                    break
                
                d = get_swap_delta(i, j, p, a, b)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    improved = True
    
    return p.tolist()
# EVOLVE-BLOCK-END
