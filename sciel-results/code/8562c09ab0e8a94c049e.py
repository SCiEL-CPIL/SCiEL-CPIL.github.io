"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _compute_rc_cc(a, b, p):
    """Precompute RC[i] = sum_k a[i,k]*b[p[i],p[k]] and CC[i] = sum_k a[k,i]*b[p[k],p[i]]."""
    n = len(a)
    bp = b[np.ix_(p, p)]
    rc = a @ bp       # rc[i] = sum_k a[i,k]*b[p[i],p[k]]
    cc = a.T @ bp     # cc[i] = sum_k a[k,i]*b[p[k],p[i]]
    return rc, cc


def _delta_swap_fast(a, b, p, rc, cc, i, j):
    """
    O(1) delta for swapping positions i and j using precomputed rc/cc vectors.
    """
    li = p[i]
    lj = p[j]
    
    # RC[i] = sum_k a[i,k]*b[li, p[k]]
    # After swap, p[i]=lj, so new RC[i] = sum_k a[i,k]*b[lj, p[k]]
    # The difference: only k=i and k=j change because p[i] and p[j] swap
    # For k not in {i,j}: b[lj,p[k]] vs b[li,p[k]] -- these are already in rc via the full sum
    # Actually, rc[i] = sum_k a[i,k]*b[li,p[k]]
    # new_rc_i = sum_k a[i,k]*b[lj,p[k]]
    # new_rc_i - rc[i] = sum_k a[i,k]*(b[lj,p[k]] - b[li,p[k]])
    # But this is O(n). Let's use the incremental approach differently.
    
    # Better: compute the exact delta using the 4 corner terms + bulk terms
    # Bulk: for k not in {i,j}:
    #   (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    # This is still O(n). 
    # 
    # Alternative: use rc/cc to get the bulk in O(1):
    # rc[i] = sum_k a[i,k]*b[li,p[k]]
    # rc[j] = sum_k a[j,k]*b[lj,p[k]]
    # 
    # After swap:
    # new_rc[i] = sum_k a[i,k]*b[lj,p[k]]  (p[i]=lj now)
    # new_rc[j] = sum_k a[j,k]*b[li,p[k]]  (p[j]=li now)
    # 
    # new_rc[i] = rc[i] - a[i,i]*b[li,li] - a[i,j]*b[li,lj] + a[i,i]*b[lj,lj] + a[i,j]*b[lj,li]
    #   Wait, that's not right. rc[i] = sum_k a[i,k]*b[li,p[k]]
    # new_rc[i] = sum_k a[i,k]*b[lj,p[k]]
    # The difference is sum_k a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    # For k=i: a[i,i]*(b[lj,li]-b[li,li])  [p[i]=li before, becomes lj]
    #   Wait no. p[i]=li before swap. After swap p[i]=lj.
    #   In rc[i] = sum_k a[i,k]*b[li,p[k]], the term k=i is a[i,i]*b[li,li]
    #   In new_rc[i] = sum_k a[i,k]*b[lj,p_new[k]], the term k=i is a[i,i]*b[lj,lj]
    #   For k=j: in rc[i], term is a[i,j]*b[li,lj]. In new_rc[i], term is a[i,j]*b[lj,li]
    #   For k not in {i,j}: p[k] unchanged, so term is a[i,k]*b[lj,p[k]] vs a[i,k]*b[li,p[k]]
    #
    # So new_rc[i] - rc[i] = a[i,i]*(b[lj,lj]-b[li,li]) + a[i,j]*(b[lj,li]-b[li,lj])
    #                        + sum_{k not in {i,j}} a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    #
    # That last sum is still O(n). This approach doesn't give O(1).
    
    # Let me just use the O(n) vectorized version but make it as fast as possible.
    li = p[i]
    lj = p[j]
    
    # The delta formula (verified):
    # delta = sum_{k not in {i,j}} (a[i,k]-a[j,k])*(b[lj,p[k]]-b[li,p[k]])
    #       + sum_{k not in {i,j}} (a[k,i]-a[k,j])*(b[p[k],lj]-b[p[k],li])
    #       + a[i,i]*(b[lj,lj]-b[li,li]) + a[i,j]*(b[lj,li]-b[li,lj])
    #       + a[j,i]*(b[li,lj]-b[lj,li]) + a[j,j]*(b[li,li]-b[lj,lj])
    
    # Use rc/cc to compute the bulk sums in O(1)?
    # rc[i] = sum_k a[i,k]*b[li,p[k]]
    # sum_{k} a[i,k]*b[lj,p[k]] = ? We don't have this precomputed.
    
    # Actually, let's precompute both row vectors:
    # For each facility i and location l, we could precompute R[i,l] = sum_k a[i,k]*b[l,p[k]]
    # But that's O(n^2) memory and O(n^3) to compute/update.
    
    # For n=150, O(n) per delta is 150 ops. Let's just make the O(n) version very tight.
    diff_row = a[i] - a[j]
    diff_col = a[:, i] - a[:, j]
    
    b_lj_p = b[lj, p]
    b_li_p = b[li, p]
    b_p_lj = b[p, lj]
    b_p_li = b[p, li]
    
    total = diff_row @ (b_lj_p - b_li_p) + diff_col @ (b_p_lj - b_p_li)
    
    total -= (a[i,i]-a[j,i])*(b[lj,li]-b[li,li])
    total -= (a[i,j]-a[j,j])*(b[lj,lj]-b[li,lj])
    total -= (a[i,i]-a[i,j])*(b[li,lj]-b[li,li])
    total -= (a[j,i]-a[j,j])*(b[lj,lj]-b[lj,li])
    
    total += a[i,i]*(b[lj,lj]-b[li,li])
    total += a[i,j]*(b[lj,li]-b[li,lj])
    total += a[j,i]*(b[li,lj]-b[lj,li])
    total += a[j,j]*(b[li,li]-b[lj,lj])
    
    return int(total)


def _local_search_2opt(a, b, p, deadline, max_passes=50):
    """First-improvement 2-opt local search with fast delta. Returns improved p and cost."""
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap_fast(a, b, p, None, None, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break  # restart inner loop after improvement
    return p, current_cost


def _initial_permutation(a, b):
    """Generate initial permutation using multiple heuristics."""
    n = len(a)
    candidates = []
    
    # Method 1: Rank matching (heavy facilities to central locations)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    candidates.append(( _qap_cost(a, b, p1), p1.copy()))
    
    # Method 2: Spectral embedding with Hungarian
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        k = min(10, n)
        idx_a = np.argsort(w_a)[-k:]
        idx_b = np.argsort(w_b)[-k:]
        X = v_a[:, idx_a]
        Y = v_b[:, idx_b]
        C = -X @ Y.T
        row_ind, col_ind = linear_sum_assignment(C)
        p2 = np.empty(n, dtype=np.int64)
        p2[row_ind] = col_ind
        candidates.append((_qap_cost(a, b, p2), p2.copy()))
    except Exception:
        pass
    
    # Method 3: Multiple random starts
    for _ in range(5):
        p_r = np.random.permutation(n).astype(np.int64)
        candidates.append((_qap_cost(a, b, p_r), p_r.copy()))
    
    # Method 4: Greedy construction
    # Place facilities one by one, choosing the location that minimizes incremental cost
    p4 = np.zeros(n, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    # Order facilities by total flow (most connected first)
    fac_order = np.argsort(-a.sum(axis=1), kind="stable")
    for idx, fi in enumerate(fac_order):
        best_loc = -1
        best_cost_loc = np.inf
        for loc in range(n):
            if used_locs[loc]:
                continue
            # Incremental cost of placing facility fi at loc
            inc = 0
            for fj in fac_order[:idx]:
                fj_loc = p4[fj]
                inc += a[fi, fj] * b[loc, fj_loc] + a[fj, fi] * b[fj_loc, loc]
            if inc < best_cost_loc:
                best_cost_loc = inc
                best_loc = loc
        p4[fi] = best_loc
        used_locs[best_loc] = True
    candidates.append((_qap_cost(a, b, p4), p4.copy()))
    
    best_cost = min(c[0] for c in candidates)
    best_p = [c[1] for c in candidates if c[0] == best_cost][0].copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin (150s limit)
    
    # Generate initial solution
    p, best_cost = _initial_permutation(a, b)
    current_cost = best_cost
    best_p = p.copy()
    
    # Phase 1: Simulated Annealing (long run)
    T = 2000.0
    T_min = 1.0
    alpha = 0.9998
    
    while T > T_min and time.time() < deadline - 40.0:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _delta_swap_fast(a, b, p, None, None, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= alpha
    
    # Phase 2: Local search polish on best solution found
    if time.time() < deadline - 15.0:
        p_ls = best_p.copy()
        p_ls, cost_ls = _local_search_2opt(a, b, p_ls, deadline - 10.0, max_passes=50)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    # Phase 3: SA restart from best if time remains
    if time.time() < deadline - 25.0:
        p2 = best_p.copy()
        T2 = 1000.0
        T2_min = 1.0
        alpha2 = 0.9998
        current_cost2 = best_cost
        
        while T2 > T2_min and time.time() < deadline - 15.0:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            delta = _delta_swap_fast(a, b, p2, None, None, i, j)
            
            if delta < 0 or np.random.random() < np.exp(-delta / T2):
                p2[i], p2[j] = p2[j], p2[i]
                current_cost2 += delta
                
                if current_cost2 < best_cost:
                    best_cost = current_cost2
                    best_p = p2.copy()
            
            T2 *= alpha2
        
        # Final local search
        if time.time() < deadline - 5.0:
            p_ls2 = best_p.copy()
            p_ls2, cost_ls2 = _local_search_2opt(a, b, p_ls2, deadline - 2.0, max_passes=30)
            if cost_ls2 < best_cost:
                best_cost = cost_ls2
                best_p = p_ls2
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
