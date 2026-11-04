"""
Memetic Genetic Algorithm (GA) for QAPLIB wil100 with Lagrangian Relaxation Initialization.
Strategy:
1. Lagrangian Relaxation: Decompose QAP into Linear Assignment + Penalty.
   Iteratively solve LAP with adjusted costs to find a high-quality surrogate permutation.
2. Use this surrogate as the elite initial solution.
3. Population-based search with vectorized O(n^2) swap delta computation for fast local search.
4. Cycle Crossover + multi-point swap mutation.
5. Tournament selection with elitism.
6. Final high-precision polish pass.
"""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b, mode='row'):
    """
    Generate an initial permutation using the Hungarian algorithm on a surrogate cost matrix.
    mode: 'row' or 'col'
    """
    n = len(a)
    if mode == 'row':
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        C = -np.outer(a_sums, b_sums)
    elif mode == 'col':
        a_sums = a.sum(axis=0)
        b_sums = b.sum(axis=0)
        C = -np.outer(a_sums, b_sums)
    else:
        C = -np.outer(a.sum(axis=1), b.sum(axis=1))
        
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _lagrangian_initial(a, b, time_limit=5.0):
    """
    Use Lagrangian Relaxation to find a high-quality initial permutation.
    Objective: min sum_{i,j} a[i,j] b[p[i], p[j]]
    Relaxation: min sum_i sum_j a[i,j] b[p[i], p[j]] + sum_i lambda_i * (something)
    
    A common approach is to fix one side or use a surrogate.
    Here we use an iterative subgradient method on the cost matrix for the LAP.
    We approximate the quadratic term with a linear term derived from the current assignment.
    """
    n = len(a)
    start_time = time.time()
    
    # Start with a simple surrogate
    p = _hungarian_initial(a, b, 'row')
    
    # We want to minimize sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let's try to construct a linear cost matrix L[i, k] such that minimizing sum L[i, p[i]] approximates QAP.
    # A standard heuristic is L[i, k] = sum_j a[i, j] * b[k, p[j]] + sum_j a[j, i] * b[p[j], k]
    # This is the gradient of the QAP cost with respect to moving facility i to location k,
    # holding other assignments fixed.
    
    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)
    
    # Iterative improvement of the surrogate
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
            
        # Compute linear cost matrix based on current p
        # L[i, k] = sum_j a[i, j] b[k, p[j]] + sum_j a[j, i] b[p[j], k]
        # Note: b is symmetric in QAP usually, but let's be general.
        # If b is symmetric, b[k, p[j]] == b[p[j], k].
        # L[i, k] = 2 * sum_j a[i, j] b[k, p[j]] ? No, the indices in a and b are coupled.
        
        # Cost = sum_i sum_j a[i, j] b[p[i], p[j]]
        # Term for i: sum_j a[i, j] b[p[i], p[j]]
        # If we move i to k, the cost change involves:
        # 1. Terms where i is the first index: sum_j a[i, j] b[k, p[j]]
        # 2. Terms where i is the second index: sum_j a[j, i] b[p[j], k]
        
        # We want to minimize the total cost.
        # Construct C_lap[i, k] = sum_j a[i, j] b[k, p[j]] + sum_j a[j, i] b[p[j], k]
        
        # Vectorized computation:
        # b_kp = b[np.ix_(range(n), p)] -> shape (n, n), row k, col j is b[k, p[j]]
        b_kp = b[:, p] # shape (n, n), b_kp[k, j] = b[k, p[j]]
        b_pk = b[p, :] # shape (n, n), b_pk[j, k] = b[p[j], k]
        
        # Sum over j for each i
        # Term 1: a @ b_kp.T ?
        # (a @ b_kp.T)[i, k] = sum_j a[i, j] b_kp[k, j] = sum_j a[i, j] b[k, p[j]]
        term1 = a @ b_kp.T
        
        # Term 2: a.T @ b_pk ?
        # (a.T @ b_pk)[i, k] = sum_j a.T[i, j] b_pk[j, k] = sum_j a[j, i] b[p[j], k]
        term2 = a.T @ b_pk
        
        C_lap = term1 + term2
        
        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(C_lap)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        
        cost_new = _qap_cost(a, b, p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new
        else:
            # If no improvement, maybe add some noise or break
            # To avoid getting stuck, we can perturb p slightly if no improvement for a few steps
            # But for initialization, we just want a good start.
            # If we don't improve, the surrogate is likely saturated.
            # We'll just break to save time for the GA.
            break
            
    return best_p, best_cost

def _compute_all_swap_deltas(a, b, p):
    """
    Compute the cost delta for swapping every pair of positions (i, j) 
    in the current permutation p.
    
    Returns a matrix D where D[i, j] is the change in cost if we swap 
    positions i and j in p.
    """
    C = b[np.ix_(p, p)]
    # Precompute symmetric part of a for reuse
    a_sym = a + a.T
    
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    # Efficient computation using broadcasting
    Term1 = S1 + S1.T - D_S1[:, None] - D_S1[None, :]
    Term2 = T1 + T1.T - D_T1[:, None] - D_T1[None, :]
    
    Raw = Term1 + Term2
    
    # Error correction term
    Error = -2.0 * a_sym * C
    
    Delta = Raw - Error
    
    # Zero out diagonal
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _local_search(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search with early termination.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        deltas = _compute_all_swap_deltas(a, b, p)
        
        # Find the best improving move
        min_idx = np.argmin(deltas)
        min_delta = deltas.flat[min_idx]
        
        if min_delta >= 0:
            break # Local minimum reached
            
        # Find indices of the best move
        i, j = np.unravel_index(min_idx, deltas.shape)
        
        # Apply swap
        p[i], p[j] = p[j], p[i]
        current_cost += min_delta
        
    return p, current_cost

def _cycle_crossover(p1, p2):
    """
    Cycle Crossover (CX).
    """
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    visited = np.zeros(n, dtype=bool)
    
    # We need to map values to positions in p2 for O(1) lookup
    pos_in_p2 = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos_in_p2[p2[i]] = i
        
    s = random.randint(0, n - 1)
    current = s
    while not visited[current]:
        visited[current] = True
        child[current] = p1[current]
        val = p1[current]
        current = pos_in_p2[val]
        
    # Fill the rest from p2
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
            
    return child

def _mutate(p, num_swaps):
    """
    Apply random swaps to the permutation.
    """
    n = len(p)
    for _ in range(num_swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # 5s buffer
    start_time = time.time()
    
    # Population parameters
    pop_size = 40
    elite_size = 4
    mutation_swaps = 2
    
    # Initialize population
    population = []
    costs = []
    
    # 1. Lagrangian Initialization (High Quality Start)
    p_lag, c_lag = _lagrangian_initial(a, b, time_limit=5.0)
    p_lag, c_lag = _local_search(a, b, p_lag, max_iters=20, time_limit=3.0, start_time=start_time)
    population.append(p_lag.copy())
    costs.append(c_lag)
    
    # 2. Hungarian Inits
    for mode in ['row', 'col']:
        if time.time() - start_time > total_deadline - 10.0:
            break
        p = _hungarian_initial(a, b, mode)
        p, c = _local_search(a, b, p, max_iters=10, time_limit=2.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    # 3. Fill rest with random + lighter local search
    while len(population) < pop_size:
        if time.time() - start_time > total_deadline - 10.0:
            break
        p = np.random.permutation(n)
        p, c = _local_search(a, b, p, max_iters=8, time_limit=1.0, start_time=start_time)
        population.append(p.copy())
        costs.append(c)
        
    best_idx = np.argmin(costs)
    best_p = population[best_idx].copy()
    best_cost = costs[best_idx]
    
    # Main GA Loop
    gen = 0
    while time.time() - start_time < total_deadline - 15.0:
        # Sort population by cost
        indices = np.argsort(costs)
        population = [population[i] for i in indices]
        costs = [costs[i] for i in indices]
        
        # Update global best
        if costs[0] < best_cost:
            best_cost = costs[0]
            best_p = population[0].copy()
            
        # Elitism
        new_population = [population[i].copy() for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p1 = population[best_cand]
            
            candidates = random.sample(range(len(population)), 3)
            best_cand = min(candidates, key=lambda x: costs[x])
            p2 = population[best_cand]
            
            # Crossover
            child = _cycle_crossover(p1, p2)
            
            # Mutation
            child = _mutate(child, mutation_swaps)
            
            # Check time remaining
            time_left = total_deadline - (time.time() - start_time)
            if time_left < 1.0:
                break
            
            # Local search
            child, c = _local_search(a, b, child, max_iters=12, time_limit=0.3, start_time=time.time())
            
            new_population.append(child)
            
        population = new_population
        costs = [_qap_cost(a, b, p) for p in population]
        
        gen += 1
    
    # Final polish phase: intensive local search on the best solution
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 5.0:
        best_p, best_cost = _local_search(a, b, best_p, max_iters=200, time_limit=time_left - 2.0, start_time=start_time)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
