"""QAP solver for tai256c: greedy construction + 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return np.sum(a * bp)

def _greedy_initial(a, b):
    """Greedy construction: place facilities one by one, choosing best location."""
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    
    # Precompute flow row sums to order facilities
    flow_sums = a.sum(axis=1)
    order = np.argsort(-flow_sums, kind='stable')
    
    # Current cost contribution: when we place facility i at location loc,
    # the incremental cost is sum over already-placed j of a[i,j]*b[loc,p[j]] + a[j,i]*b[p[j],loc]
    # We track this incrementally.
    
    for idx in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if placed[loc]:
                continue
            # Incremental cost of placing facility idx at location loc
            # = sum_{j placed} a[idx,j]*b[loc,p[j]] + a[j,idx]*b[p[j],loc]
            # We can vectorize this
            placed_mask = placed.copy()
            placed_mask[loc] = False  # loc is not placed yet
            if not placed_mask.any():
                inc = 0.0
            else:
                placed_locs = p[placed_mask]
                # a[idx, placed_facilities] * b[loc, placed_locs] + a[placed_facilities, idx] * b[placed_locs, loc]
                placed_facilities = np.where(placed_mask)[0]
                inc = np.sum(a[idx, placed_facilities] * b[loc, placed_locs]) + \
                      np.sum(a[placed_facilities, idx] * b[placed_locs, loc])
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[idx] = best_loc
        placed[best_loc] = True
    
    return p

def _local_search_2opt(a, b, p, time_limit):
    """2-opt local search: swap pairs of facilities, accept if cost improves."""
    n = len(a)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= time_limit:
                    break
                # Compute cost delta of swapping p[i] and p[j]
                pi, pj = p[i], p[j]
                # Delta = -a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi] 
                #       + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
                #       + sum_{k != i,j} [a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] 
                #                         + a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi]
                #                         - a[i,k]*b[pi,p[k]] - a[k,i]*b[p[k],pi]
                #                         - a[j,k]*b[pj,p[k]] - a[k,j]*b[p[k],pj]]
                
                # Vectorized computation
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                others = np.where(mask)[0]
                others_locs = p[others]
                
                # Terms involving i and j with others
                # For facility i: was at pi, now at pj
                # For facility j: was at pj, now at pi
                
                # Old contributions with others:
                # a[i,k]*b[pi,p[k]] + a[k,i]*b[p[k],pi] for all k
                # a[j,k]*b[pj,p[k]] + a[k,j]*b[p[k],pj] for all k
                # New contributions with others:
                # a[i,k]*b[pj,p[k]] + a[k,i]*b[p[k],pj] for all k
                # a[j,k]*b[pi,p[k]] + a[k,j]*b[p[k],pi] for all k
                
                old_i = np.sum(a[i, others] * b[pi, others_locs]) + np.sum(a[others, i] * b[others_locs, pi])
                old_j = np.sum(a[j, others] * b[pj, others_locs]) + np.sum(a[others, j] * b[others_locs, pj])
                new_i = np.sum(a[i, others] * b[pj, others_locs]) + np.sum(a[others, i] * b[others_locs, pj])
                new_j = np.sum(a[j, others] * b[pi, others_locs]) + np.sum(a[others, j] * b[others_locs, pi])
                
                # Direct i-j terms
                old_ij = a[i, j] * b[pi, pj] + a[j, i] * b[pj, pi]
                new_ij = a[i, j] * b[pj, pi] + a[j, i] * b[pi, pj]
                
                delta = (new_i + new_j + new_ij) - (old_i + old_j + old_ij)
                
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Greedy initial solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local search with time limit
    remaining = deadline - time.time()
    if remaining > 1.0:
        p_ls, cost_ls = _local_search_2opt(a, b, p, remaining - 1.0)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
