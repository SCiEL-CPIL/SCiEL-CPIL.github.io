"""
Strategy: Iterated Local Search (ILS) with 2-swap neighborhood and random restarts.
This is a structurally different approach from Simulated Annealing.
Instead of a single trajectory with temperature-based acceptance, ILS maintains a single
best solution and periodically perturbs it to escape local optima, then applies a fast
local search (2-swap) to refine the perturbed solution. This is often more effective
for QAP than SA because it focuses computational effort on intensifying promising regions
rather than exploring broadly with a cooling schedule.

Key differences from the current SA program:
1. No temperature parameter or Metropolis acceptance.
2. Uses a dedicated local search phase (2-swap) after each perturbation.
3. Perturbation is a fixed number of random swaps, not tied to a cooling schedule.
4. The local search is deterministic (first-improvement or best-improvement) rather than stochastic.
5. Initialization uses a simple rank-matching heuristic (sort facilities by flow degree, sort locations by distance degree) which is fast and often competitive.
"""

import numpy as np
import time
import random
import sys

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j, a_sym):
    """
    Compute the change in cost when swapping facilities at positions i and j.
    p[i] is location of facility i.
    a_sym = a + a.T
    """
    x = i
    y = j
    Lx = p[x]
    Ly = p[y]
    
    if Lx == Ly:
        return 0
        
    # diff_b[k] = b[Ly, p[k]] - b[Lx, p[k]]
    diff_b = b[Ly, p] - b[Lx, p]
    
    # coeff[k] = a_sym[x, k] - a_sym[y, k]
    coeff = a_sym[x, :] - a_sym[y, :]
    
    delta = np.dot(coeff, diff_b)
    return delta

def _local_search_2swap(a, b, p, a_sym, max_iter=10000, time_limit=None, start_time=None):
    """
    Perform local search using 2-swap moves with first-improvement strategy.
    Returns the improved permutation and cost.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved:
        improved = False
        iter_count += 1
        if iter_count > max_iter:
            break
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Try all pairs (i, j) with i < j
        for i in range(n):
            if improved:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j, a_sym)
                if delta < 0:
                    # Accept the move
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    break
                    # After a swap, the cost of other pairs changes, so we restart the scan
                    # This is a simple first-improvement strategy
    
    return p, current_cost

def _rank_matching_init(a, b):
    """
    Initialize permutation using rank-matching heuristic.
    Sort facilities by their total flow (row sum of A).
    Sort locations by their total distance (row sum of B).
    Match the k-th most flowy facility to the k-th most central location.
    """
    n = len(a)
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Facility "importance": total flow
    fac_weights = a_sym.sum(axis=1)
    # Location "centrality": total distance (lower is more central)
    loc_weights = b_sym.sum(axis=1)
    
    # Sort facilities in descending order of flow
    fac_order = np.argsort(-fac_weights, kind="stable")
    # Sort locations in ascending order of distance (most central first)
    loc_order = np.argsort(loc_weights, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
        
    return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    a_sym = a + a.T
    
    # 1. Initialization
    p = _rank_matching_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Initial Local Search
    p, current_cost = _local_search_2swap(a, b, p, a_sym, time_limit=time_limit, start_time=start_time)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. Iterated Local Search Loop
    # Perturb the best solution, then apply local search
    # Keep the best solution found so far
    
    while time.time() - start_time < time_limit:
        # Perturb the best solution
        # Number of swaps for perturbation: typically sqrt(n) or a small fraction of n
        # For n=256, sqrt(n) ~ 16. Let's use 10-20 swaps.
        num_swaps = random.randint(10, 20)
        p = best_p.copy()
        for _ in range(num_swaps):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        
        # Apply local search to the perturbed solution
        p, current_cost = _local_search_2swap(a, b, p, a_sym, time_limit=time_limit, start_time=start_time)
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Check time
        if time.time() - start_time >= time_limit:
            break
    
    return best_p.tolist()
