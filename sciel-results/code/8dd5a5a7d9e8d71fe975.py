"""QAP solver for tai256c with multi-start 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_matching_init(a, b):
    """Rank-matching: heavy flow rows to central location rows."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(len(a), dtype=np.int64)
    p[heavy] = central
    return p


def _random_init(n, rng):
    """Random permutation."""
    p = np.arange(n, dtype=np.int64)
    rng.shuffle(p)
    return p


def _local_search_2opt(a, b, p, deadline, max_rounds=50):
    """2-opt local search with vectorized delta computation.
    
    Returns (improved_p, improved_cost, converged).
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    converged = False
    round_num = 0
    
    # Precompute b_rows and b_cols
    b_rows = b[p, :]  # shape (n, n), b_rows[i] = b[p[i], :]
    b_cols = b[:, p]  # shape (n, n), b_cols[i] = b[:, p[i]]
    
    while not converged and round_num < max_rounds:
        round_num += 1
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, current_cost, False
            
            # Compute delta for swapping i with all j > i
            b_diff = b_rows - b_rows[i]  # shape (n, n)
            a_diff = a[i] - a  # shape (n, n)
            term2 = np.sum(a_diff * b_diff, axis=1)  # shape (n,)
            
            a_T = a.T
            a_T_diff = a_T[i] - a_T  # shape (n, n)
            b_cols_diff = b_cols.T - b_cols[i]  # shape (n, n)
            term3 = np.sum(a_T_diff * b_cols_diff, axis=1)  # shape (n,)
            
            term1 = (a[i] - a.T[i]) * (b_rows[:, i] - b_rows[i, :])  # shape (n,)
            
            delta = term1 + term2 + term3  # shape (n,)
            
            # Find best j > i with negative delta
            best_j = -1
            best_delta = 0.0
            for j in range(i + 1, n):
                if delta[j] < best_delta:
                    best_delta = delta[j]
                    best_j = j
            
            if best_j >= 0 and best_delta < -1e-9:
                j = best_j
                # Perform the swap
                p[i], p[j] = p[j], p[i]
                current_cost += int(best_delta)
                
                # Update b_rows and b_cols
                b_rows[i], b_rows[j] = b_rows[j], b_rows[i]
                b_cols[:, [i, j]] = b_cols[:, [j, i]]
                
                improved = True
                break
        
        if not improved:
            converged = True
    
    return p, current_cost, converged


def _perturb(p, rng, strength=0.1):
    """Perturb a permutation by swapping a random number of pairs."""
    n = len(p)
    p = p.copy()
    num_swaps = max(2, int(strength * n))
    for _ in range(num_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Initial solution: rank matching
    best_p = _rank_matching_init(a, b)
    best_cost = _qap_cost(a, b, best_p)
    
    # Also try a random init
    rand_p = _random_init(n, rng)
    rand_cost = _qap_cost(a, b, rand_p)
    if rand_cost < best_cost:
        best_cost = rand_cost
        best_p = rand_p
    
    # Iterated Local Search
    stagnation = 0
    max_stagnation = 30
    
    while time.time() < deadline:
        # Local search from current best
        p_ls, cost_ls, converged = _local_search_2opt(a, b, best_p, deadline)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls
            stagnation = 0
        else:
            stagnation += 1
        
        if stagnation >= max_stagnation:
            # Try a fresh random restart if stuck too long
            fresh_p = _random_init(n, rng)
            fresh_cost = _qap_cost(a, b, fresh_p)
            if fresh_cost < best_cost:
                best_cost = fresh_cost
                best_p = fresh_p
                stagnation = 0
            else:
                break
        
        # Perturb the current best to create a new starting point
        perturbed_p = _perturb(best_p, rng, strength=0.08)
        
        # Check if we have time for another iteration
        if time.time() > deadline - 10.0:
            break
    
    # Final polish: one more local search if time permits
    if time.time() < deadline - 5.0:
        best_p, best_cost, _ = _local_search_2opt(a, b, best_p, deadline)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
