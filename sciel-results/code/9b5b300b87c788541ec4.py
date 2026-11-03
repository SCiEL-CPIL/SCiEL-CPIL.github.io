"""Search-based solver for QAPLIB wil100: ILSA + SA with precomputed row sums for fast deltas."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _precompute(a, b):
    """Precompute row sums of A and B for fast O(n) swap delta evaluation."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    return a_row_sums, b_row_sums


def _swap_delta(a, b, a_rs, b_rs, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    Uses precomputed row sums for O(n) evaluation.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    # Create mask for all indices except i and j
    if n <= 2:
        other_mask = np.zeros(n, dtype=bool)
    else:
        other_mask = np.ones(n, dtype=bool)
        other_mask[i] = False
        other_mask[j] = False
    
    if not np.any(other_mask):
        other_mask = None
    
    # Direct terms (i,j pairs)
    delta = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
             a[j, j] * (b[pi, pi] - b[pj, pj]) +
             a[i, j] * (b[pj, pi] - b[pi, pj]) +
             a[j, i] * (b[pi, pj] - b[pj, pi]))
    
    if other_mask is not None and np.any(other_mask):
        # Cross terms with other indices
        # For l in other: (a[i,l] - a[j,l]) * (b[pj, p[l]] - b[pi, p[l]])
        # For k in other: (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
        
        # We can use precomputed row sums to speed this up
        # sum_l a[i,l]*b[pj,p[l]] - sum_l a[j,l]*b[pj,p[l]]
        # = b[pj, :] @ a[i, :]^T - b[pj, :] @ a[j, :]^T (but restricted to other indices)
        
        # Actually, let's just do the direct computation since n=100 is small
        p_other = p[other_mask]
        
        term1 = np.sum((a[i, other_mask] - a[j, other_mask]) * (b[pj, p_other] - b[pi, p_other]))
        term2 = np.sum((a[other_mask, i] - a[other_mask, j]) * (b[p_other, pj] - b[p_other, pi]))
        
        delta += term1 + term2
    
    return delta


def _greedy_initial(a, b):
    """Create an initial permutation using rank-matching heuristic."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search(a, b, a_rs, b_rs, p, time_limit):
    """Perform best-improvement local search with O(n) delta evaluation."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Find best swap
        best_delta = 0
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, a_rs, b_rs, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0 and best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_p, best_cost


def _sa_polish(a, b, a_rs, b_rs, p, time_limit):
    """Simulated annealing polish phase."""
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    t_start = time.time()
    t_end = time_limit
    
    # Cooling schedule
    t0 = 1000.0
    t_final = 0.1
    
    while time.time() - t_start < t_end:
        # Temperature
        elapsed = time.time() - t_start
        frac = elapsed / t_end
        temp = t0 * (t_final / t0) ** frac
        
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, a_rs, b_rs, p, i, j)
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost


def _perturb(p, k):
    """Perturb permutation by k random swaps."""
    n = len(p)
    p = p.copy()
    for _ in range(k):
        i, j = np.random.choice(n, 2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    deadline = 110.0  # Safety margin before 120s limit
    start = time.time()
    
    np.random.seed(42)
    
    a_rs, b_rs = _precompute(a, b)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy 1: Greedy initial solution
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    best_p = p_greedy.copy()
    best_cost = cost_greedy
    
    # Strategy 2: ILSA - Iterated Local Search with SA polish
    num_restarts = 50
    max_iter = 0
    
    for restart in range(num_restarts):
        if time.time() - start > deadline - 15:
            break
        
        # Choose starting point
        if restart == 0:
            p_curr = best_p.copy()
        else:
            # Perturb the best solution
            k = 3 + restart % 5
            p_curr = _perturb(best_p, k)
        
        # Local search
        time_remaining = deadline - (time.time() - start) - 5
        if time_remaining < 2:
            break
        
        # Allocate time between LS and SA
        ls_time = min(3.0, time_remaining * 0.4)
        sa_time = min(2.0, time_remaining * 0.3)
        
        p_ls, cost_ls = _local_search(a, b, a_rs, b_rs, p_curr.copy(), ls_time)
        
        # SA polish
        p_sa, cost_sa = _sa_polish(a, b, a_rs, b_rs, p_ls.copy(), sa_time)
        
        if cost_sa < best_cost:
            best_cost = cost_sa
            best_p = p_sa.copy()
        
        # Check time
        if time.time() - start > deadline - 20:
            break
    
    # Final long SA run from best solution
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 10:
        p_final, cost_final = _sa_polish(a, b, a_rs, b_rs, best_p.copy(), time_remaining - 5)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
