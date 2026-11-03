"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    Uses vectorized sum-minus-exclusion for speed.
    """
    x = p[i]
    y = p[j]
    
    # Direct pair contribution
    delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
    
    # Vectorized sum over all k
    V1 = b[y, p]
    V2 = b[x, p]
    V3 = b[p, y]
    V4 = b[p, x]
    
    T1 = a[i, :] * (V1 - V2)
    T2 = a[:, i] * (V3 - V4)
    T3 = a[j, :] * (V2 - V1)
    T4 = a[:, j] * (V4 - V3)
    
    S = np.sum(T1 + T2 + T3 + T4)
    
    # Subtract terms for k=i and k=j
    # T(i)
    T_i = (a[i, i] * (b[y, x] - b[x, x]) + 
           a[i, i] * (b[x, y] - b[x, x]) + 
           a[j, i] * (b[x, x] - b[y, x]) + 
           a[i, j] * (b[x, x] - b[x, y]))
           
    # T(j)
    T_j = (a[i, j] * (b[y, y] - b[x, y]) + 
           a[j, i] * (b[y, y] - b[y, x]) + 
           a[j, j] * (b[x, y] - b[y, y]) + 
           a[j, j] * (b[y, x] - b[y, y]))
           
    delta += S - T_i - T_j
    
    return int(delta)


def _greedy_init(a, b, n):
    """Greedy initialization based on row sums."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_init(n):
    """Random permutation initialization."""
    return np.random.permutation(n)


def _tabu_search(a, b, p, time_limit, tabu_tenure=30, max_iters=None, sample_size=1500):
    """Tabu search with 1-swap neighborhood."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_list = {}
    iteration = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if max_iters is not None and iteration >= max_iters:
            break
        
        iteration += 1
        
        # Best improving 1-swap
        best_swap_i, best_swap_j = -1, -1
        best_delta = 0
        
        # Sample a subset of swaps for speed
        actual_sample_size = min(sample_size, n * n // 4)
        
        if actual_sample_size < n * n:
            # Generate random pairs using randint for speed
            i_arr = np.random.randint(0, n, actual_sample_size)
            j_arr = np.random.randint(0, n, actual_sample_size)
            
            # Filter i != j
            mask = i_arr != j_arr
            i_arr = i_arr[mask]
            j_arr = j_arr[mask]
            
            for k in range(len(i_arr)):
                i = int(i_arr[k])
                j = int(j_arr[k])
                
                key = (i, j) if i < j else (j, i)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                
                delta = _swap_delta(a, b, p, i, j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap_i, best_swap_j = i, j
        else:
            # Full neighborhood (unlikely for n=256)
            for i in range(n):
                for j in range(i+1, n):
                    key = (i, j)
                    if key in tabu_list and tabu_list[key] > iteration:
                        continue
                    delta = _swap_delta(a, b, p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap_i, best_swap_j = i, j
        
        if best_swap_i == -1:
            break
        
        # Perform the swap
        i, j = best_swap_i, best_swap_j
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iteration + tabu_tenure
        
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Clean up expired tabu entries occasionally
        if iteration % 1000 == 0:
            expired = [k for k, v in tabu_list.items() if v <= iteration]
            for k in expired:
                del tabu_list[k]
    
    return best_p, best_cost





def _solve_tai256c(a, b):
    n = len(a)
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    inits.append(_greedy_init(a, b, n))
    for _ in range(4):
        inits.append(_random_init(n))
    
    best_p = None
    best_cost = np.inf
    
    # Phase 2: Quick Tabu Search on each initialization to find a good basin
    # Allocate ~5s per init
    quick_time = 5.0
    
    for i, p_init in enumerate(inits):
        if time.time() >= deadline - 30:
            break
        
        p_tabu, cost_tabu = _tabu_search(a, b, p_init, time_limit=quick_time, sample_size=500)
        
        if cost_tabu < best_cost:
            best_cost = cost_tabu
            best_p = p_tabu.copy()
    
    # Phase 3: Long Tabu Search from the best quick solution
    if best_p is not None and time.time() < deadline - 5:
        polish_time = deadline - time.time() - 5
        p_polished, cost_polished = _tabu_search(a, b, best_p, time_limit=polish_time, sample_size=2000)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # Fallback if no time was left or error
    if best_p is None:
        p_greedy = _greedy_init(a, b, n)
        best_p, best_cost = _tabu_search(a, b, p_greedy, time_limit=min(10.0, deadline - time.time() - 5))
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
