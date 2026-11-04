"""
Strategy: Parallel Island Model with Tabu Search for QAP.

Structural differences from the current ILS program:
1. Instead of a single sequential trajectory, we run 4 independent "islands" in parallel threads.
2. Each island uses Tabu Search (TS) instead of deterministic 2-swap local search.
   - TS maintains a tabu list of recently swapped facility pairs, preventing immediate reversals.
   - This allows accepting non-improving moves (if not tabu), enabling escape from local optima.
3. Periodic migration: every K iterations, the best solution from each island is shared with a neighbor island.
4. Initialization: each island starts from a slightly perturbed rank-matching solution to ensure diversity.
5. The parallel structure allows more total search effort within the time budget, as threads run concurrently.
"""

import numpy as np
import time
import random
import threading
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
        
    diff_b = b[Ly, p] - b[Lx, p]
    coeff = a_sym[x, :] - a_sym[y, :]
    
    delta = np.dot(coeff, diff_b)
    return delta

def _rank_matching_init(a, b, seed=None):
    """
    Initialize permutation using rank-matching heuristic with optional perturbation.
    """
    n = len(a)
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    fac_weights = a_sym.sum(axis=1)
    loc_weights = b_sym.sum(axis=1)
    
    fac_order = np.argsort(-fac_weights, kind="stable")
    loc_order = np.argsort(loc_weights, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    for k in range(n):
        p[fac_order[k]] = loc_order[k]
    
    # Apply small random perturbation for diversity
    for _ in range(rng.randint(5, 15)):
        i, j = rng.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
        
    return p

def _tabu_search(a, b, p, a_sym, time_limit, start_time, island_id, migration_queue, migration_interval=200):
    """
    Run Tabu Search on a single island.
    Returns the best permutation and cost found.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Tabu list: stores (i, j) pairs that are tabu
    tabu_list = []
    tabu_tenure = 20  # Number of iterations a move stays tabu
    
    iter_count = 0
    no_improve_count = 0
    max_no_improve = 500  # Restart if stuck
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Check for migration
        if iter_count % migration_interval == 0 and migration_queue is not None:
            # Send our best to the queue
            migration_queue.put((island_id, best_p.copy(), best_cost))
            # Try to receive a better solution from another island
            try:
                src_id, src_p, src_cost = migration_queue.get_nowait()
                if src_id != island_id and src_cost < best_cost:
                    # Accept the better solution
                    p = src_p.copy()
                    current_cost = src_cost
                    best_p = p.copy()
                    best_cost = current_cost
                    tabu_list = []  # Reset tabu list
            except:
                pass
        
        # Find the best non-tabu move
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Sample a subset of pairs for efficiency (full O(n^2) is too slow per iteration)
        # For n=256, n*(n-1)/2 = 32640 pairs. We sample ~5000 pairs per iteration.
        sample_size = min(5000, n * (n - 1) // 2)
        pairs = []
        for _ in range(sample_size):
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            pairs.append((i, j))
        
        # Remove duplicates
        pairs = list(set(pairs))
        
        for i, j in pairs:
            # Check if this move is tabu
            is_tabu = False
            for t_iter, t_i, t_j in tabu_list:
                if (t_i == i and t_j == j) or (t_i == j and t_j == i):
                    is_tabu = True
                    break
            
            if is_tabu:
                continue
            
            delta = _swap_delta(a, b, p, i, j, a_sym)
            
            # Aspiration criterion: accept tabu move if it improves best
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
        
        # If no non-tabu improving move, try the best tabu move (aspiration)
        if best_i == -1:
            # Find best move overall, ignoring tabu status
            best_delta = 0
            for i, j in pairs:
                delta = _swap_delta(a, b, p, i, j, a_sym)
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
            
            if best_i == -1:
                # No improving move at all, restart with perturbation
                for _ in range(10):
                    i, j = random.sample(range(n), 2)
                    p[i], p[j] = p[j], p[i]
                current_cost = _qap_cost(a, b, p)
                tabu_list = []
                no_improve_count = 0
                continue
        
        # Apply the best move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update tabu list
        tabu_list.append((iter_count, best_i, best_j))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck for too long, perturb and reset
        if no_improve_count > max_no_improve:
            for _ in range(random.randint(5, 20)):
                i, j = random.sample(range(n), 2)
                p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            tabu_list = []
            no_improve_count = 0
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    a_sym = a + a.T
    
    # Number of islands
    num_islands = 4
    
    # Create migration queues
    migration_queues = [None] * num_islands
    for i in range(num_islands):
        migration_queues[i] = threading.Queue()
    
    # Store results from each island
    results = [None] * num_islands
    result_locks = [threading.Lock() for _ in range(num_islands)]
    
    # Define island worker function
    def island_worker(island_id):
        # Each island starts from a perturbed rank-matching solution
        p_init = _rank_matching_init(a, b, seed=island_id * 1000 + 42)
        
        # Run tabu search
        best_p, best_cost = _tabu_search(
            a, b, p_init, a_sym, 
            time_limit, start_time, 
            island_id, 
            migration_queues[island_id],
            migration_interval=200
        )
        
        with result_locks[island_id]:
            results[island_id] = (best_p, best_cost)
    
    # Start all islands in parallel
    threads = []
    for i in range(num_islands):
        t = threading.Thread(target=island_worker, args=(i,))
        t.start()
        threads.append(t)
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Find the best result across all islands
    best_p = None
    best_cost = float('inf')
    
    for i in range(num_islands):
        if results[i] is not None:
            p, cost = results[i]
            if cost < best_cost:
                best_cost = cost
                best_p = p
    
    # If no result (shouldn't happen), fall back to rank matching
    if best_p is None:
        best_p = _rank_matching_init(a, b, seed=0)
    
    return best_p.tolist()
