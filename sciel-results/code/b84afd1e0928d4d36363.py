"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j, a_sym=None):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    p is a numpy array where p[k] is the location assigned to facility k.
    """
    if a_sym is None:
        a_sym = a + a.T
    
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta


def _hungarian_step(a, b, p, a_sym=None):
    """
    Perform a global reassignment using the Hungarian algorithm.
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    B = b[:, p]
    
    c = a_sym @ B.T
    
    diag_a = np.diag(a)
    for i in range(n):
        if diag_a[i] != 0:
            c[i, :] -= 2 * diag_a[i] * B[:, i]
    
    row_ind, col_ind = linear_sum_assignment(c)
    p_new = np.empty(n, dtype=np.int64)
    p_new[row_ind] = col_ind
    
    return p_new


def _local_search_2opt(a, b, p, time_limit, a_sym=None, max_passes=10):
    """
    Perform 2-opt local search on permutation p.
    Returns the improved permutation and its cost.
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    break
                delta = _swap_delta(a, b, best_p, i, j, a_sym)
                if delta < 0:
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
    
    return best_p, best_cost


def _greedy_init(a, b, a_sym):
    """
    Greedy construction: iteratively place facilities to minimize incremental cost.
    """
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    unplaced = list(range(n))
    placed = []
    
    # Start with the facility that has the largest row sum in a
    first = int(np.argmax(a_row_sums_cache))
    # Place it at the location that minimizes sum_k a[first,k] * b[loc, ?] -- but we don't know others yet
    # For greedy, place at location with smallest row sum in b (most central)
    loc = int(np.argmin(b_row_sums_cache))
    p[first] = loc
    placed.append(first)
    unplaced.remove(first)
    
    while unplaced:
        # For each unplaced facility, compute best location
        best_fac = -1
        best_loc = -1
        best_cost = np.inf
        for fac in unplaced:
            # For each available location, compute incremental cost
            # Incremental cost of placing fac at loc:
            # sum_{k in placed} (a[fac,k] + a[k,fac]) * b[loc, p[k]]
            # + 2 * a[fac,fac] * b[loc, loc] if fac==fac... no, just the cross terms
            inc_costs = np.zeros(n)
            for k in placed:
                inc_costs[p[k]] += (a[fac, k] + a[k, fac])  # will multiply by b[loc, p[k]]
            # Actually: inc_cost(loc) = sum_{k in placed} (a[fac,k] + a[k,fac]) * b[loc, p[k]]
            # We can compute this as: for each placed k, add (a[fac,k]+a[k,fac]) * b[:, p[k]] to inc_costs
            # But that's O(|placed| * n) per facility, which is O(n^2) per step, O(n^3) total
            # For n=256, that's ~16M operations per step, 256 steps = ~4B. Too slow.
            
            # Faster: precompute for all unplaced facilities at once
            # Let's just do a simpler approach: pick facility with max row sum, place at min b row sum among available
            pass
        
        # Simpler greedy: sort facilities by row sum desc, place at locations sorted by row sum asc
        # This is the rank matching, which we already have.
        # Let's do a slightly better greedy: for each step, try a few candidate facilities and locations
        # and pick the one with lowest incremental cost.
        
        # To keep it fast, just pick the next facility (by row sum) and best available location
        fac = unplaced[0]
        # Find best location for this facility
        inc = np.zeros(n)
        for k in placed:
            inc += (a[fac, k] + a[k, fac]) * b[:, p[k]]
        # Only consider unoccupied locations
        occupied = set(placed_locs)
        best_loc = -1
        best_val = np.inf
        for loc in range(n):
            if loc not in occupied:
                val = inc[loc]
                if val < best_val:
                    best_val = val
                    best_loc = loc
        p[fac] = best_loc
        placed.append(fac)
        placed_locs.append(best_loc)
        unplaced.remove(fac)
    
    return p


def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Generate initial candidate using rank matching
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    best_p = p0.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Main loop: alternate between Hungarian global reassignment and 2-opt polishing
    # with multiple diverse starting points
    num_starts = 5
    start_candidates = [p0]
    
    # Create diverse starting points
    for s in range(1, num_starts):
        p_s = p0.copy()
        # Random perturbation
        for _ in range(10 + s * 5):
            i, j = random.sample(range(n), 2)
            p_s[i], p_s[j] = p_s[j], p_s[i]
        start_candidates.append(p_s)
    
    # Evaluate each starting point with Hungarian + 2-opt
    for idx, p_start in enumerate(start_candidates):
        if time.time() - start_time > time_limit - 15.0:
            break
        
        # Hungarian step
        p_hung = _hungarian_step(a, b, p_start, a_sym)
        
        # 2-opt polishing
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_hung, time_limit - 10.0, a_sym, max_passes=3
        )
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
    
    # If time remains, do more iterations of Hungarian + 2-opt from best
    while time.time() - start_time < time_limit - 10.0:
        p_hung = _hungarian_step(a, b, best_p, a_sym)
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_hung, time_limit - 5.0, a_sym, max_passes=3
        )
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
        else:
            # Perturb and retry
            p_pert = best_p.copy()
            for _ in range(8):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            p_polished2, cost_polished2 = _local_search_2opt(
                a, b, p_pert, time_limit - 5.0, a_sym, max_passes=2
            )
            if cost_polished2 < best_cost:
                best_cost = cost_polished2
                best_p = p_polished2.copy()
    
    # Final polish with more time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5.0:
        best_p, best_cost = _local_search_2opt(
            a, b, best_p, time_limit - 1.0, a_sym, max_passes=10
        )
    
    return best_p.tolist()


def solve(a, b):
    global a_row_sums_cache, b_row_sums_cache
    a_row_sums_cache = a.sum(axis=1)
    b_row_sums_cache = b.sum(axis=1)
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
