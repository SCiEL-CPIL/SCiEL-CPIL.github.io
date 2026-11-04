"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    n = len(p)
    total = 0
    for i in range(n):
        total += np.dot(a[i, :], b[p[i], :])
    return total


def _local_search(a, b, p, deadline=None):
    """
    Perform local search by trying all pairs (k,l) and accepting the best improving swap.
    p is a list of ints.
    Returns the improved permutation and its cost.
    """
    n = len(p)
    p = list(p)
    current_cost = _compute_cost(a, b, p)
    
    improved = True
    passes = 0
    
    while improved and passes < 50:
        improved = False
        best_delta = 0
        best_k = -1
        best_l = -1
        
        for k in range(n):
            for l in range(k + 1, n):
                pk = p[k]
                pl = p[l]
                
                # Compute delta for swapping p[k] and p[l]
                # Diagonal terms
                delta = a[k, k] * (b[pl, pl] - b[pk, pk])
                delta += a[l, l] * (b[pk, pk] - b[pl, pl])
                
                # Cross terms
                delta += a[k, l] * (b[pl, pk] - b[pk, pl])
                delta += a[l, k] * (b[pk, pl] - b[pl, pk])
                
                # Sum over j not in {k,l} of (a[k,j] - a[l,j]) * (b[pl,p[j]] - b[pk,p[j]])
                # = sum_j (a[k,j] - a[l,j]) * b[pl,p[j]] - sum_j (a[k,j] - a[l,j]) * b[pk,p[j]]
                #   - (j=k term) - (j=l term)
                
                # Precompute: for each facility f, sum_j a[f,j] * b[g, p[j]] for various g
                # This is expensive. Let me just use numpy for the sum.
                
                # For j terms:
                # Create a vector v[j] = (a[k,j] - a[l,j]) * (b[pl, p[j]] - b[pk, p[j]])
                # Sum over all j, then subtract j=k and j=l terms
                
                diff_a = a[k, :] - a[l, :]
                diff_b_pl = b[pl, p] - b[pk, p]  # This is b[pl, p[j]] - b[pk, p[j]] for each j
                j_sum = np.dot(diff_a, diff_b_pl)
                
                # Subtract j=k term: (a[k,k] - a[l,k]) * (b[pl, p[k]] - b[pk, p[k]])
                j_sum -= (a[k, k] - a[l, k]) * (b[pl, pk] - b[pk, pk])
                # Subtract j=l term: (a[k,l] - a[l,l]) * (b[pl, p[l]] - b[pk, p[l]])
                j_sum -= (a[k, l] - a[l, l]) * (b[pl, pl] - b[pk, pl])
                
                delta += j_sum
                
                # Sum over i not in {k,l} of (a[i,k] - a[i,l]) * (b[p[i], pl] - b[p[i], pk])
                # = sum_i (a[i,k] - a[i,l]) * b[p[i], pl] - sum_i (a[i,k] - a[i,l]) * b[p[i], pk]
                #   - (i=k term) - (i=l term)
                
                diff_a_col = a[:, k] - a[:, l]
                # b[p[i], pl] - b[p[i], pk] for each i
                diff_b_pl_col = b[p, pl] - b[p, pk]
                i_sum = np.dot(diff_a_col, diff_b_pl_col)
                
                # Subtract i=k term: (a[k,k] - a[k,l]) * (b[p[k], pl] - b[p[k], pk])
                i_sum -= (a[k, k] - a[k, l]) * (b[pk, pl] - b[pk, pk])
                # Subtract i=l term: (a[l,k] - a[l,l]) * (b[p[l], pl] - b[p[l], pk])
                i_sum -= (a[l, k] - a[l, l]) * (b[pl, pl] - b[pl, pk])
                
                delta += i_sum
                
                if delta < best_delta:
                    best_delta = delta
                    best_k = k
                    best_l = l
            
            # Check deadline
            if deadline is not None and time.time() > deadline:
                break
        
        if deadline is not None and time.time() > deadline:
            break
        
        if best_k >= 0:
            p[best_k], p[best_l] = p[best_l], p[best_k]
            current_cost += best_delta
            improved = True
            passes += 1
    
    return p, current_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial permutation: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    best_p = list(p)
    best_cost = _compute_cost(a, b, p)
    
    # Local search from initial
    p, cost = _local_search(a, b, p, deadline=deadline)
    if cost < best_cost:
        best_cost = cost
        best_p = list(p)
    
    # Check if we have time for restarts
    if time.time() < deadline - 10.0:
        random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution
            p = list(best_p)
            num_swaps = max(1, n // 20)
            for _ in range(num_swaps):
                k = random.randint(0, n - 1)
                l = random.randint(0, n - 1)
                if k != l:
                    p[k], p[l] = p[l], p[k]
            
            # Local search
            p, cost = _local_search(a, b, p, deadline=deadline)
            if cost < best_cost:
                best_cost = cost
                best_p = list(p)
    
    return best_p

# EVOLVE-BLOCK-END
