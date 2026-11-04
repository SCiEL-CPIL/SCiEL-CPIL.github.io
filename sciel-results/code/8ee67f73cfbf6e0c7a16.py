"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return np.sum(a * b_p)

def _init_rank_matching(a, b):
    """Rank-matching initialization: sort facilities by flow centrality, locations by distance centrality."""
    n = len(a)
    # Facility centrality: sum of flows from/to each facility
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    # Location centrality: sum of distances to/from each location
    dist_centrality = b.sum(axis=0) + b.sum(axis=1)
    
    heavy = np.argsort(-flow_centrality, kind="stable")
    central = np.argsort(dist_centrality, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _init_greedy(a, b):
    """Greedy construction: place facilities one by one in order of flow centrality."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    order = np.argsort(-flow_centrality, kind="stable")
    
    p = np.full(n, -1, dtype=np.int64)
    used_locs = np.zeros(n, dtype=bool)
    
    for fac in order:
        # For each unused location, compute the cost contribution of placing fac there
        # Cost contribution = sum_{k placed} a[fac,k]*b[loc,p[k]] + a[k,fac]*b[p[k],loc]
        # + a[fac,fac]*b[loc,loc] (but diagonal is usually 0 or small)
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if used_locs[loc]:
                continue
            # Compute incremental cost
            # We need to sum over all already-placed facilities
            placed_mask = p >= 0
            if not np.any(placed_mask):
                cost = 0
            else:
                placed_facs = np.where(placed_mask)[0]
                # a[fac, placed_facs] * b[loc, p[placed_facs]] + a[placed_facs, fac] * b[p[placed_facs], loc]
                cost = np.dot(a[fac, placed_facs], b[loc, p[placed_facs]]) + \
                       np.dot(a[placed_facs, fac], b[p[placed_facs], loc])
            if cost < best_cost:
                best_cost = cost
                best_loc = loc
        p[fac] = best_loc
        used_locs[best_loc] = True
    
    return p

def _swap_cost_delta_fast(a, b, p, i, j, row_sums, col_sums):
    """
    Fast swap cost delta using precomputed row/col contributions.
    
    We maintain:
      row_sums[i] = sum_k a[i,k] * b[p[i], p[k]]  (cost contribution of facility i to rows)
      col_sums[i] = sum_k a[k,i] * b[p[k], p[i]]  (cost contribution of facility i to cols)
    
    Actually, let's use a different approach. Precompute for each pair of locations (li, lj)
    the vector d[l] = b[l, p] - b[lj, p] etc. This is complex.
    
    Alternative: Just compute the 4 dot products but with pre-extracted vectors.
    """
    li = p[i]
    lj = p[j]
    
    # Pre-extract the needed b rows/cols as contiguous arrays
    b_li_p = b[li, p]  # b[li, p[k]] for all k
    b_lj_p = b[lj, p]  # b[lj, p[k]] for all k
    b_p_li = b[p, li]  # b[p[k], li] for all k
    b_p_lj = b[p, lj]  # b[p[k], lj] for all k
    
    # Difference vectors
    diff_row = b_lj_p - b_li_p   # b[lj, p[k]] - b[li, p[k]]
    diff_col = b_p_lj - b_p_li   # b[p[k], lj] - b[p[k], li]
    
    a_i = a[i, :]
    a_col_i = a[:, i]
    a_j = a[j, :]
    a_col_j = a[:, j]
    
    # Full sums (including k=i and k=j)
    term1_all = np.dot(a_i, diff_row)
    term2_all = np.dot(a_col_i, diff_col)
    term3_all = np.dot(a_j, -diff_row)  # b[li,p[k]] - b[lj,p[k]] = -(b[lj,p[k]] - b[li,p[k]])
    term4_all = np.dot(a_col_j, -diff_col)
    
    # Subtract k=i and k=j terms
    # term1: k=i: a[i,i]*(b[lj,li]-b[li,li]), k=j: a[i,j]*(b[lj,lj]-b[li,lj])
    term1 = term1_all - a[i,i]*(b[lj,li]-b[li,li]) - a[i,j]*(b[lj,lj]-b[li,lj])
    
    # term2: k=i: a[i,i]*(b[li,lj]-b[li,li]), k=j: a[j,i]*(b[lj,lj]-b[lj,li])
    term2 = term2_all - a[i,i]*(b[li,lj]-b[li,li]) - a[j,i]*(b[lj,lj]-b[lj,li])
    
    # term3: k=i: a[j,i]*(b[li,li]-b[lj,li]), k=j: a[j,j]*(b[li,lj]-b[lj,lj])
    term3 = term3_all - a[j,i]*(b[li,li]-b[lj,li]) - a[j,j]*(b[li,lj]-b[lj,lj])
    
    # term4: k=i: a[i,j]*(b[li,li]-b[li,lj]), k=j: a[j,j]*(b[lj,li]-b[lj,lj])
    term4 = term4_all - a[i,j]*(b[li,li]-b[li,lj]) - a[j,j]*(b[lj,li]-b[lj,lj])
    
    # Diagonal/cross terms
    delta_diag = (a[i,i]*(b[lj,lj]-b[li,li]) + 
                  a[i,j]*(b[lj,li]-b[li,lj]) + 
                  a[j,i]*(b[li,lj]-b[lj,li]) + 
                  a[j,j]*(b[li,li]-b[lj,lj]))
    
    return delta_diag + term1 + term2 + term3 + term4

def _sa_run(a, b, p_init, max_time, rng, t_start, t_end, cooling_rate):
    """Run simulated annealing for max_time seconds, return best permutation and cost."""
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Batch processing for efficiency
    batch_size = 5000
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        # Determine how many steps we can do in remaining time
        remaining = max_time - elapsed
        if remaining < 0.1:
            break
        
        # Do a batch of steps
        for _ in range(batch_size):
            if time.time() - start_time > max_time:
                break
            
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i == j:
                continue
            
            delta = _swap_cost_delta_fast(a, b, p, i, j, None, None)
            
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                current_cost += delta
            else:
                if t_start > t_end:
                    # Use log comparison to avoid underflow
                    log_prob = -delta / t_start
                    if log_prob > -700:  # exp(-700) is essentially 0
                        if math.log(rng.random()) < log_prob:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
            
            # Cool down
            t_start *= cooling_rate
            if t_start < t_end:
                t_start = t_end
            
            # Track best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    max_total_time = 115.0
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and SA runs
    best_p = None
    best_cost = np.inf
    
    # Strategy: Do several SA runs with different seeds/initializations
    # Use rank matching for first, then perturbed versions
    
    # Run 1: Rank matching init
    p_init = _init_rank_matching(a, b)
    t_start = 30000.0
    t_end = 1.0
    cooling_rate = 0.99995
    
    # Estimate how much time per run
    # We'll do 3 runs of ~35s each
    time_per_run = 35.0
    
    for run in range(3):
        if time.time() - start_time > max_total_time - 2.0:
            break
        
        if run == 0:
            p_init = _init_rank_matching(a, b)
        elif run == 1:
            # Perturb the best solution found so far
            if best_p is not None:
                p_init = best_p.copy()
                # Randomly swap 10% of elements
                n_swaps = max(5, n // 10)
                for _ in range(n_swaps):
                    i = rng.integers(0, n)
                    j = rng.integers(0, n)
                    p_init[i], p_init[j] = p_init[j], p_init[i]
            else:
                p_init = _init_rank_matching(a, b)
        else:
            # Random restart with some structure
            p_init = _init_rank_matching(a, b)
            # Apply random swaps
            n_swaps = n // 5
            for _ in range(n_swaps):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                p_init[i], p_init[j] = p_init[j], p_init[i]
        
        # Adjust temperature for subsequent runs
        if run == 0:
            t_s = 30000.0
            t_e = 1.0
            cr = 0.99995
        elif run == 1:
            t_s = 10000.0
            t_e = 1.0
            cr = 0.99998
        else:
            t_s = 20000.0
            t_e = 1.0
            cr = 0.99997
        
        p_result, cost_result = _sa_run(a, b, p_init, time_per_run, rng, t_s, t_e, cr)
        
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
    
    # Final local improvement: hill climbing on the best solution
    if best_p is not None and time.time() - start_time < max_total_time - 1.0:
        p = best_p.copy()
        current_cost = best_cost
        improved = True
        while improved and time.time() - start_time < max_total_time - 0.5:
            improved = False
            for i in range(n):
                if time.time() - start_time > max_total_time - 0.5:
                    break
                for j in range(i+1, n):
                    if time.time() - start_time > max_total_time - 0.5:
                        break
                    delta = _swap_cost_delta_fast(a, b, p, i, j, None, None)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
