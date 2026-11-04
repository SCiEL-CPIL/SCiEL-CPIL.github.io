"""
QAP wil100 solver: Greedy Construction + 2-Exchange Local Search.

Strategy:
1. Build an initial permutation greedily: at each step, place the unassigned facility
   at the unassigned location that minimizes the marginal increase in total cost.
2. Polish with a fast 2-exchange local search using vectorized O(n) delta computation.
3. Multi-start: run greedy construction with different random perturbations of the
   facility ordering, then local-search each result, keeping the best.
"""
import numpy as np
import time

def _qap_cost(a, b, p):
    """Exact integer cost."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _delta_swap(a, b, p, i, j):
    """
    O(n) vectorized delta for swapping facilities i and j in permutation p.
    Returns integer delta.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct pair term
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k (vectorized)
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _greedy_construct(a, b, facility_order):
    """
    Greedily construct a permutation by placing facilities in the given order.
    At each step, choose the unassigned location that minimizes marginal cost.
    
    facility_order: array of facility indices in the order to place them.
    Returns: permutation p of length n.
    """
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    unassigned_locs = set(range(n))
    
    # For efficiency, track which facilities are already placed
    placed = np.zeros(n, dtype=bool)
    
    # We'll compute the marginal cost incrementally.
    # When placing facility i at location l, the marginal cost is:
    # sum_{j placed} a[i,j]*b[l, p[j]] + sum_{j placed} a[j,i]*b[p[j], l]
    # + a[i,i]*b[l,l]  (diagonal)
    # We also need to account for interactions with already-placed facilities.
    
    # Precompute nothing; just iterate. For n=100 this is O(n^3) total which is fine.
    
    for idx, fac in enumerate(facility_order):
        if placed[fac]:
            continue
        best_loc = -1
        best_marginal = np.inf
        
        for loc in unassigned_locs:
            # Compute marginal cost of placing facility 'fac' at 'loc'
            marginal = 0.0
            # Diagonal
            marginal += a[fac, fac] * b[loc, loc]
            # Interactions with already placed facilities
            for j in range(n):
                if placed[j] and j != fac:
                    marginal += a[fac, j] * b[loc, p[j]]
                    marginal += a[j, fac] * b[p[j], loc]
            
            if marginal < best_marginal:
                best_marginal = marginal
                best_loc = loc
        
        p[fac] = best_loc
        placed[fac] = True
        unassigned_locs.remove(best_loc)
    
    return p

def _greedy_construct_fast(a, b, facility_order):
    """
    Faster greedy construction using vectorized marginal cost computation.
    """
    n = len(a)
    p = np.full(n, -1, dtype=np.int64)
    placed = np.zeros(n, dtype=bool)
    
    for fac in facility_order:
        if placed[fac]:
            continue
        
        # Vectorized marginal cost for all unassigned locations
        # marginal[loc] = a[fac,fac]*b[loc,loc] + sum_{j placed} (a[fac,j]*b[loc,p[j]] + a[j,fac]*b[p[j],loc])
        
        # Get indices of placed facilities
        placed_idx = np.where(placed)[0]
        
        if len(placed_idx) == 0:
            # First facility: just diagonal
            marginals = a[fac, fac] * b[np.arange(n), np.arange(n)]
        else:
            # b[loc, p[j]] for all loc and all placed j
            # p[placed_idx] gives the locations of placed facilities
            placed_locs = p[placed_idx]  # shape (k,)
            
            # a[fac, placed_idx] * b[:, placed_locs] -> shape (n, k)
            # Sum over placed facilities
            term1 = a[fac, placed_idx] * b[:, placed_locs]  # (n, k)
            # a[placed_idx, fac] * b[placed_locs, :] -> shape (k, n), transpose to (n, k)
            term2 = a[placed_idx, fac] * b[placed_locs, :]  # (k, n)
            
            marginals = a[fac, fac] * b[np.arange(n), np.arange(n)] + term1.sum(axis=1) + term2.sum(axis=0)
        
        # Mask assigned locations
        assigned_mask = (p == -1)  # unassigned locations
        marginals_masked = marginals.copy()
        marginals_masked[~assigned_mask] = np.inf
        
        best_loc = np.argmin(marginals_masked)
        p[fac] = best_loc
        placed[fac] = True
    
    return p

def _local_search_2x(a, b, p, time_limit):
    """
    Best-improvement 2-exchange local search with vectorized delta.
    Repeatedly find the best improving swap until no improvement or time limit.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Scan all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = j, i  # Note: swap i and j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            break  # Local optimum
    
    return best_p, best_cost

def _first_improvement_ls(a, b, p, time_limit):
    """
    First-improvement 2-exchange local search (faster per pass, more passes).
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break  # First improvement, restart scan
            if improved:
                break
        
        if not improved:
            break
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy 1: Greedy construction with natural order
    t0 = time.time()
    p_greedy = _greedy_construct_fast(a, b, np.arange(n))
    cost_greedy = _qap_cost(a, b, p_greedy)
    best_p = p_greedy.copy()
    best_cost = cost_greedy
    
    # Polish with local search
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5:
        ls_time = min(10.0, time_remaining * 0.3)
        p_ls, cost_ls = _local_search_2x(a, b, best_p.copy(), ls_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Strategy 2: Multi-start greedy with different facility orderings
    time_remaining = deadline - (time.time() - start)
    max_starts = 20  # Will be limited by time
    
    for s in range(max_starts):
        time_remaining = deadline - (time.time() - start)
        if time_remaining < 8.0:
            break
        
        # Create a perturbed facility ordering
        if s == 0:
            # Already done natural order
            continue
        
        if s % 3 == 1:
            # Random permutation
            order = np.random.permutation(n)
        elif s % 3 == 2:
            # Reverse natural order
            order = np.arange(n - 1, -1, -1)
        else:
            # Partially shuffled: keep first half, shuffle second half
            order = np.arange(n)
            half = n // 2
            order[half:] = np.random.permutation(order[half:])
        
        # Greedy construction
        p_g = _greedy_construct_fast(a, b, order)
        cost_g = _qap_cost(a, b, p_g)
        
        if cost_g < best_cost:
            best_cost = cost_g
            best_p = p_g.copy()
        
        # Quick local search polish
        time_remaining = deadline - (time.time() - start)
        if time_remaining > 5.0:
            ls_time = min(3.0, time_remaining * 0.2)
            p_ls, cost_ls = _first_improvement_ls(a, b, p_g.copy(), ls_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    # Strategy 3: Final deep local search on best solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 3.0:
        ls_time = min(15.0, time_remaining - 2.0)
        p_final, cost_final = _local_search_2x(a, b, best_p.copy(), ls_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()
    
    # Strategy 4: One more pass with first-improvement if time remains
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 3.0:
        fi_time = min(10.0, time_remaining - 2.0)
        p_fi, cost_fi = _first_improvement_ls(a, b, best_p.copy(), fi_time)
        if cost_fi < best_cost:
            best_cost = cost_fi
            best_p = p_fi.copy()
    
    return best_p.tolist()
