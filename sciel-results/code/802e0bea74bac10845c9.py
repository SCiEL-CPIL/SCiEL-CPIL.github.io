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


def _tabu_search(a, b, p, time_limit, a_sym=None, tabu_len=50, max_iter=20000):
    """
    Tabu search with 2-opt neighborhood.
    Uses first-improvement with tabu list.
    """
    n = len(a)
    if a_sym is None:
        a_sym = a + a.T
    
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    best_p = current_p.copy()
    best_cost = current_cost
    
    # Tabu list: store (i, j) pairs and their expiration iteration
    tabu_list = {}
    tabu_len = 50
    
    iter_count = 0
    no_improve_count = 0
    start_time = time.time()
    
    while iter_count < max_iter:
        if time.time() > time_limit:
            break
        
        found_move = False
        # Randomly shuffle the order of i for diversity
        i_order = np.random.permutation(n)
        
        for i in i_order:
            if time.time() > time_limit:
                break
            if found_move:
                break
            j_order = np.random.permutation(n)
            for j in j_order:
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in tabu_list and tabu_list[key] > iter_count:
                    continue
                
                delta = _swap_delta(a, b, current_p, i, j, a_sym)
                
                # Aspiration: accept if it improves best, even if tabu
                if delta < 0 or (key in tabu_list and tabu_list[key] <= iter_count and current_cost + delta < best_cost):
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                    current_cost += delta
                    tabu_list[key] = iter_count + tabu_len
                    found_move = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    break
                elif time.time() > time_limit:
                    break
        
        iter_count += 1
        if no_improve_count > 200:
            break
    
    return best_p, best_cost


def _solve_tai256c(a, b, time_limit=110.0):
    n = len(a)
    start_time = time.time()
    
    # Precompute
    a_sym = a + a.T
    
    # Generate initial candidate using rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Multi-start: run several local searches from different starting points
    # First, polish the initial solution
    best_p, best_cost = _local_search_2opt(a, b, best_p, time_limit - 5.0, a_sym, max_passes=5)
    
    # Now run iterated local search with diverse restarts
    # Keep the best solution found
    while time.time() - start_time < time_limit - 10.0:
        # Create a perturbed version of the best solution
        p_pert = best_p.copy()
        
        # Random perturbation: swap several pairs
        num_swaps = random.randint(5, 15)
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
        
        # Polish the perturbed solution
        p_polished, cost_polished = _local_search_2opt(
            a, b, p_pert, time_limit - 5.0, a_sym, max_passes=5
        )
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
        
        # Occasionally use tabu search for deeper exploration
        if time.time() - start_time < time_limit - 30.0 and random.random() < 0.3:
            p_tabu, cost_tabu = _tabu_search(
                a, b, best_p, time_limit - 10.0, a_sym, tabu_len=50, max_iter=5000
            )
            if cost_tabu < best_cost:
                best_cost = cost_tabu
                best_p = p_tabu.copy()
        
        # Occasionally use Hungarian reassignment
        if time.time() - start_time < time_limit - 30.0 and random.random() < 0.2:
            p_hung = _hungarian_step(a, b, best_p, a_sym)
            p_polished2, cost_polished2 = _local_search_2opt(
                a, b, p_hung, time_limit - 5.0, a_sym, max_passes=5
            )
            if cost_polished2 < best_cost:
                best_cost = cost_polished2
                best_p = p_polished2.copy()
    
    # Final polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3.0:
        best_p, best_cost = _local_search_2opt(
            a, b, best_p, time_limit - 1.0, a_sym, max_passes=5
        )
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
