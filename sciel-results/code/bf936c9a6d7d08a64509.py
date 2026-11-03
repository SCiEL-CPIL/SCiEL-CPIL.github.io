"""QAP wil100 solver: Hungarian surrogate init + vectorized SA + 2-opt polish."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    delta = cost_after - cost_before.
    """
    pi, pj = p[i], p[j]
    # Terms involving i or j change.
    # cost_before contribution from pairs (i, *), (*, i), (j, *), (*, j)
    # We can compute delta efficiently:
    # For all k != i, j:
    #   a[i,k]*(b[pj,p[k]] - b[pi,p[k]]) + a[k,i]*(b[p[k],pj] - b[p[k],pi])
    #   a[j,k]*(b[pi,p[k]] - b[pj,p[k]]) + a[k,j]*(b[p[k],pi] - b[p[k],pj])
    # Plus the (i,j) and (j,i) terms:
    #   a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
    #   = (a[i,j] - a[j,i]) * (b[pj,pi] - b[pi,pj])
    
    # Vectorized computation
    mask = np.ones(len(a), dtype=bool)
    mask[i] = False
    mask[j] = False
    pk = p[mask]
    
    # a[i, mask] * (b[pj, pk] - b[pi, pk])
    term1 = a[i, mask] @ (b[pj, pk] - b[pi, pk])
    term2 = a[mask, i] @ (b[pk, pj] - b[pk, pi])
    term3 = a[j, mask] @ (b[pi, pk] - b[pj, pk])
    term4 = a[mask, j] @ (b[pk, pi] - b[pk, pj])
    term5 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(term1 + term2 + term3 + term4 + term5)


def _hungarian_init(a, b):
    """
    Formulate a linear surrogate: match facilities to locations based on
    flow row sums vs distance row sums, using Hungarian algorithm on a
    cost matrix derived from centrality products.
    """
    n = len(a)
    # Row sums of flow matrix and distance matrix
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    # Build a cost matrix: we want high-flow facilities at low-distance locations.
    # Use a rank-based or product-based cost.
    # Simple approach: cost[i, j] = flow_sums[i] * dist_sums[j]
    # Minimize this with Hungarian.
    cost_matrix = np.outer(flow_sums, dist_sums)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _sa_search(a, b, p, time_budget):
    """
    Vectorized Simulated Annealing with 2-opt style moves (single swaps).
    Uses deadline guard.
    """
    n = len(a)
    p = p.copy().astype(np.int64)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for speed: we'll use random swaps
    # Temperature schedule: geometric cooling
    T_start = max(1.0, current_cost * 0.01)
    T_end = 1e-6
    # Estimate number of iterations we can afford
    # Each iteration is roughly O(n) due to swap_delta
    # Let's just loop until time runs out
    
    start_time = time.time()
    deadline = start_time + time_budget
    safety_margin = 5.0  # seconds before actual deadline
    
    rng = np.random.default_rng(42)
    
    # Adaptive temperature: cool based on iteration count estimate
    # But we don't know iteration count in advance, so use time-based cooling
    # T(t) = T_start * (T_end/T_start)^(t / total_time)
    total_time = time_budget
    
    it = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break
        
        # Compute temperature based on elapsed fraction
        frac = min(1.0, elapsed / total_time)
        T = T_start * (T_end / T_start) ** frac
        
        # Pick two random distinct indices
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0:
            # Accept
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Metropolis acceptance
            if T > 1e-10:
                # Accept with probability exp(-delta/T)
                # Use log to avoid overflow
                log_prob = -delta / T
                if log_prob > 0:
                    # Always accept if delta < 0, but we're in else so delta >= 0
                    pass
                if np.log(rng.random()) < log_prob:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
        
        it += 1
        # Throttle: check time every 1000 iterations
        if it % 1000 == 0:
            if time.time() - start_time > deadline - safety_margin:
                break
    
    return best_p, best_cost


def _two_opt_polish(a, b, p, time_budget):
    """
    Local search: try all pairs of swaps, accept improving moves.
    Repeat until no improvement or time runs out.
    """
    n = len(a)
    p = p.copy().astype(np.int64)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + time_budget
    safety_margin = 2.0
    
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_pair = (-1, -1)
        
        # Try all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (i, j)
        
        if best_pair[0] >= 0:
            i, j = best_pair
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta
            improved = True
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        if time.time() - start_time > deadline - safety_margin:
            break
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    total_deadline = 150.0  # wall-clock limit
    safety_margin = 10.0  # return before limit
    start_time = time.time()
    
    # Phase 1: Hungarian init (fast, < 1s)
    p = _hungarian_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    elapsed = time.time() - start_time
    if elapsed > total_deadline - safety_margin:
        return best_p.tolist()
    
    # Phase 2: Simulated Annealing
    sa_budget = min(60.0, total_deadline - safety_margin - elapsed - 20.0)
    if sa_budget > 5.0:
        p_sa, cost_sa = _sa_search(a, b, p, sa_budget)
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
    
    elapsed = time.time() - start_time
    if elapsed > total_deadline - safety_margin:
        return best_p.tolist()
    
    # Phase 3: 2-opt polish
    polish_budget = max(1.0, total_deadline - safety_margin - elapsed)
    p_polish, cost_polish = _two_opt_polish(a, b, best_p, polish_budget)
    if cost_polish < best_cost:
        best_cost = cost_polish
        best_p = p_polish.copy()
    
    return best_p.tolist()
