"""Combined QAP solver for tai256c: Vectorized 2-opt Local Search + Multi-start ILS."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP using vectorized delta calculation.
    p: array of length n, p[i] is the location assigned to facility i.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    # The cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # Let A_sym = a + a.T.
    # Delta for swapping i and j:
    # We swap locations li=p[i] and lj=p[j].
    # The change in cost comes from terms involving i or j.
    # Term(i,j) = a[i,j]b[li,lj] + a[j,i]b[lj,li]
    # Term(i,i) = a[i,i]b[li,li]
    # Term(j,j) = a[j,j]b[lj,lj]
    # Terms with k: a[i,k]b[li,p[k]] + a[k,i]b[p[k],li] + a[j,k]b[lj,p[k]] + a[k,j]b[p[k],lj]
    
    # Vectorized approach for a fixed i:
    # We want to find j > i that minimizes the delta.
    # Let's derive the vectorized formula for delta(i, j) for all j.
    #
    # Cost = sum_{u,v} a[u,v] D[u,v] where D[u,v] = b[p[u], p[v]].
    # Swap i and j.
    # New D' differs only in rows/cols i and j.
    # D'[i, i] = D[j, j]
    # D'[j, j] = D[i, i]
    # D'[i, j] = D[j, i]
    # D'[j, i] = D[i, j]
    # D'[i, k] = D[j, k] for k != i, j
    # D'[k, i] = D[k, j] for k != i, j
    # D'[j, k] = D[i, k] for k != i, j
    # D'[k, j] = D[k, i] for k != i, j
    #
    # Delta = sum_{u,v} a[u,v] (D'[u,v] - D[u,v])
    #
    # Terms involving i or j:
    # u=i, v=i: a[i,i](D[j,j]-D[i,i])
    # u=j, v=j: a[j,j](D[i,i]-D[j,j])
    # u=i, v=j: a[i,j](D[j,i]-D[i,j])
    # u=j, v=i: a[j,i](D[i,j]-D[j,i])
    # u=i, v=k (k!=i,j): a[i,k](D[j,k]-D[i,k])
    # u=k, v=i (k!=i,j): a[k,i](D[k,j]-D[k,i])
    # u=j, v=k (k!=i,j): a[j,k](D[i,k]-D[j,k])
    # u=k, v=j (k!=i,j): a[k,j](D[k,i]-D[k,j])
    #
    # Grouping terms for a fixed i, and varying j:
    # Let's define vectors for a fixed i.
    # A_row_i = a[i, :]
    # A_col_i = a[:, i]
    # A_row_j = a[j, :]
    # A_col_j = a[:, j]
    #
    # This is complex to fully vectorize over j for all i simultaneously in one shot without O(n^3) memory.
    # Instead, we iterate i, and vectorize over j.
    
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # We need to calculate delta(i, j) for all j in js.
            # Let's use the derived formula:
            # Delta(i, j) = 
            #   (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            #   + (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            #   + sum_{k != i, j} [ a[i,k](D[j,k]-D[i,k]) + a[k,i](D[k,j]-D[k,i]) 
            #                       + a[j,k](D[i,k]-D[j,k]) + a[k,j](D[k,i]-D[k,j]) ]
            
            # Let's simplify the sum over k.
            # Sum_k a[i,k](D[j,k]-D[i,k]) = a[i, :] dot (D[j, :] - D[i, :])
            # Sum_k a[k,i](D[k,j]-D[k,i]) = a[:, i] dot (D[:, j] - D[:, i])
            # Sum_k a[j,k](D[i,k]-D[j,k]) = a[j, :] dot (D[i, :] - D[j, :])
            # Sum_k a[k,j](D[k,i]-D[k,j]) = a[:, j] dot (D[:, i] - D[:, j])
            
            # Note: The sums include k=i and k=j. We need to exclude them or adjust.
            # Let's compute the full dot products first.
            
            # Term 1: Sum_k a[i,k](D[j,k]-D[i,k])
            # = a[i, :] @ (D[j, :] - D[i, :])
            # This includes k=i and k=j.
            # k=i: a[i,i](D[j,i]-D[i,i])
            # k=j: a[i,j](D[j,j]-D[i,j])
            
            # Term 2: Sum_k a[k,i](D[k,j]-D[k,i])
            # = a[:, i] @ (D[:, j] - D[:, i])
            # k=i: a[i,i](D[i,j]-D[i,i])
            # k=j: a[j,i](D[j,j]-D[j,i])
            
            # Term 3: Sum_k a[j,k](D[i,k]-D[j,k])
            # = a[j, :] @ (D[i, :] - D[j, :])
            # k=i: a[j,i](D[i,i]-D[j,i])
            # k=j: a[j,j](D[i,j]-D[j,j])
            
            # Term 4: Sum_k a[k,j](D[k,i]-D[k,j])
            # = a[:, j] @ (D[:, i] - D[:, j])
            # k=i: a[i,j](D[i,i]-D[i,j])
            # k=j: a[j,j](D[j,i]-D[j,j])
            
            # Let's sum these 4 terms.
            # Let S = T1 + T2 + T3 + T4.
            # We want to exclude k=i and k=j from the sums, then add back the specific diagonal/cross terms.
            
            # Vectorized calculation for all j in js:
            # D_i = D[i, :]
            # D_js = D[js, :]  # shape (m, n) where m = n-i-1
            # D_col_i = D[:, i]
            # D_col_js = D[:, js] # shape (n, m)
            
            D_i = D[i, :]
            D_js = D[js, :]
            D_col_i = D[:, i]
            D_col_js = D[:, js]
            
            a_i = a[i, :]
            a_js = a[js, :]
            a_col_i = a[:, i]
            a_col_js = a[:, js]
            
            # T1_j = a[i, :] @ (D[j, :] - D[i, :]) = a_i @ D[j, :] - a_i @ D_i
            # a_i @ D[j, :] for all j in js is (a_i @ D_js.T).T ? No.
            # D_js is (m, n). a_i is (n,).
            # We want sum_k a_i[k] * D_js[:, k]. This is a_i @ D_js.T -> shape (m,).
            T1 = a_i @ D_js.T - (a_i @ D_i)
            
            # T2_j = a[:, i] @ (D[:, j] - D[:, i]) = a_col_i @ D[:, j] - a_col_i @ D_col_i
            # a_col_i @ D[:, j] for all j in js.
            # D_col_js is (n, m). a_col_i is (n,).
            # We want sum_k a_col_i[k] * D[k, j]. This is a_col_i @ D_col_js -> shape (m,).
            T2 = a_col_i @ D_col_js - (a_col_i @ D_col_i)
            
            # T3_j = a[j, :] @ (D[i, :] - D[j, :]) = a[j, :] @ D_i - a[j, :] @ D[j, :]
            # a[j, :] @ D_i for all j in js is (D_i @ a_js.T).T ?
            # a_js is (m, n). D_i is (n,).
            # We want sum_k a_js[:, k] * D_i[k]. This is a_js @ D_i -> shape (m,).
            T3 = a_js @ D_i - (a_js @ D_js) # Wait, a_js @ D_js is sum_k a_js[:, k] * D_js[:, k]. This is element-wise dot? No.
            # a_js @ D_js is matrix multiplication (m, n) @ (m, n) -> invalid.
            # We need sum_k a[j, k] * D[j, k]. This is np.sum(a_js * D_js, axis=1).
            T3 = a_js @ D_i - np.sum(a_js * D_js, axis=1)
            
            # T4_j = a[:, j] @ (D[:, i] - D[:, j]) = a[:, j] @ D_col_i - a[:, j] @ D[:, j]
            # a[:, j] @ D_col_i for all j in js.
            # a_col_js is (n, m). D_col_i is (n,).
            # We want sum_k a[k, j] * D[k, i]. This is D_col_i @ a_col_js -> shape (m,).
            # a[:, j] @ D[:, j] is sum_k a[k, j] * D[k, j]. This is np.sum(a_col_js * D_col_js, axis=0).
            T4 = D_col_i @ a_col_js - np.sum(a_col_js * D_col_js, axis=0)
            
            S = T1 + T2 + T3 + T4
            
            # Now subtract the k=i and k=j contributions from S, because we only want k != i, j.
            # S includes:
            # k=i:
            # T1_i = a[i,i](D[j,i]-D[i,i])
            # T2_i = a[i,i](D[i,j]-D[i,i])
            # T3_i = a[j,i](D[i,i]-D[j,i])
            # T4_i = a[i,j](D[i,i]-D[i,j])
            # Sum_k=i = a[i,i](D[j,i]-2D[i,i]+D[i,j]) + a[j,i](D[i,i]-D[j,i]) + a[i,j](D[i,i]-D[i,j])
            
            # k=j:
            # T1_j = a[i,j](D[j,j]-D[i,j])
            # T2_j = a[j,i](D[j,j]-D[j,i])
            # T3_j = a[j,j](D[i,j]-D[j,j])
            # T4_j = a[j,j](D[j,i]-D[j,j])
            # Sum_k=j = a[i,j](D[j,j]-D[i,j]) + a[j,i](D[j,j]-D[j,i]) + a[j,j](D[i,j]-2D[j,j]+D[j,i])
            
            # This is getting messy. Let's use the simpler explicit formula for the correction terms.
            # The formula derived earlier:
            # Delta = S_corrected + diag_term + cross_term
            # Where S_corrected is sum_{k!=i,j} ...
            
            # Let's just compute the full S (which includes i and j) and then subtract the i and j parts explicitly.
            
            # Part from k=i in S:
            # T1_i = a[i,i] * (D[js, i] - D[i, i])
            # T2_i = a[i,i] * (D[i, js] - D[i, i])  # Note D[i, js] is row i, cols js
            # T3_i = a[js, i] * (D[i, i] - D[js, i]) # a[j, i]
            # T4_i = a[i, js] * (D[i, i] - D[i, js]) # a[i, j]
            
            k_i_part = (a[i, i] * (D[js, i] - D[i, i]) + 
                        a[i, i] * (D[i, js] - D[i, i]) + 
                        a[js, i] * (D[i, i] - D[js, i]) + 
                        a[i, js] * (D[i, i] - D[i, js]))
            
            # Part from k=j in S:
            # T1_j = a[i, js] * (D[js, js] - D[i, js]) # a[i, j] * (D[j, j] - D[i, j])
            # T2_j = a[js, i] * (D[js, js] - D[js, i]) # a[j, i] * (D[j, j] - D[j, i])
            # T3_j = a[js, js] * (D[i, js] - D[js, js]) # a[j, j] * (D[i, j] - D[j, j])
            # T4_j = a[js, js] * (D[js, i] - D[js, js]) # a[j, j] * (D[j, i] - D[j, j])
            
            k_j_part = (a[i, js] * (D[js, js] - D[i, js]) + 
                        a[js, i] * (D[js, js] - D[js, i]) + 
                        a[js, js] * (D[i, js] - D[js, js]) + 
                        a[js, js] * (D[js, i] - D[js, js]))
            
            S_corrected = S - k_i_part - k_j_part
            
            # Now add the direct swap terms (u=i,v=i; u=j,v=j; u=i,v=j; u=j,v=i)
            # diag_term = (a[i,i] - a[j,j]) * (D[j,j] - D[i,i])
            # cross_term = (a[i,j] - a[j,i]) * (D[j,i] - D[i,j])
            
            diag_term = (a[i, i] - a[js, js]) * (D[js, js] - D[i, i])
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S_corrected + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
                # We do not break the inner loop, we continue to next i?
                # Standard 2-opt: if we swap, we might want to restart i=0.
                # But for speed, we often just continue. Let's restart i=0 to be safe and thorough.
                break
        
        if improved:
            continue
        else:
            break
            
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _rank_init(a, b, reverse=False):
    """Rank-matching initialization."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind='stable')
    central = np.argsort(b.sum(axis=1), kind='stable')
    if reverse:
        central = central[::-1]
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # Leave 5s buffer for return
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multi-start Local Search
    inits = []
    inits.append(_rank_init(a, b, reverse=False))
    inits.append(_rank_init(a, b, reverse=True))
    inits.append(_hungarian_init(a, b))
    
    # Add 2 random inits
    for _ in range(2):
        inits.append(np.random.permutation(n).astype(np.int64))
    
    # Allocate time for each initial local search
    # We have ~145s. 5 inits. Let's give each 20s. Total 100s.
    # Remaining 45s for ILS.
    
    time_per_ls = 20.0
    
    for init_p in inits:
        if time.time() - start_time > time_budget - 30: # Leave 30s for ILS
            break
        p, cost = _local_search(a, b, init_p, time_per_ls, start_time)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    if best_p is None:
        best_p = inits[0]
        best_cost = _qap_cost(a, b, best_p)
        
    # Phase 2: Iterated Local Search (ILS)
    # Use remaining time for perturb-and-restart.
    # We perturb the best solution found so far, then run local search.
    # If it improves, update best.
    
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 2:
        # Adaptive perturbation
        if consecutive_no_improve > 10:
            perturb_k = min(15, base_perturb + consecutive_no_improve // 2)
        else:
            perturb_k = base_perturb
            
        p_pert = best_p.copy()
        for _ in range(perturb_k):
            i, j = np.random.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Allocate time for this LS run
        # Give it a chunk, but not too much if we are close to deadline.
        duration = (time_budget - 2) - (time.time() - start_time)
        if duration < 1:
            break
            
        # Cap the time for a single LS run in ILS to allow for more restarts
        # If we have lots of time, allow longer.
        ls_time_limit = min(duration, 10.0)
        
        p_ls, cost_ls = _local_search(a, b, p_pert, ls_time_limit, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
    # Final Polish
    # One last run with all remaining time to settle.
    if time.time() - start_time < time_budget - 1:
        p_final, cost_final = _local_search(a, b, best_p, time_budget, start_time)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
            
    return best_p.tolist()
# EVOLVE-BLOCK-END
