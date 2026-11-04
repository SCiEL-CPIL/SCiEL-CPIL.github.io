"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    b_p = b[np.ix_(p, p)]
    return int(np.sum(a * b_p))

def _swap_delta(a, b, p, i, j):
    """
    Compute cost delta for swapping p[i] and p[j].
    """
    li = p[i]
    lj = p[j]
    
    # Create mask excluding i and j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # Term 1: sum_{k!=i,j} a[i,k]*(b[lj,p[k]]-b[li,p[k]])
    t1 = np.dot(a[i, mask], b[lj, p[mask]] - b[li, p[mask]])
    
    # Term 2: sum_{k!=i,j} a[k,i]*(b[p[k],lj]-b[p[k],li])
    t2 = np.dot(a[mask, i], b[p[mask], lj] - b[p[mask], li])
    
    # Term 3: sum_{k!=i,j} a[j,k]*(b[li,p[k]]-b[lj,p[k]])
    t3 = np.dot(a[j, mask], b[li, p[mask]] - b[lj, p[mask]])
    
    # Term 4: sum_{k!=i,j} a[k,j]*(b[p[k],li]-b[p[k],lj])
    t4 = np.dot(a[mask, j], b[p[mask], li] - b[p[mask], lj])
    
    # Cross terms
    t5 = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    
    # Diagonal terms
    t6 = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
    
    return int(t1 + t2 + t3 + t4 + t5 + t6)

def _init_rank_matching(a, b):
    """Rank-matching initialization."""
    n = len(a)
    flow_centrality = a.sum(axis=0) + a.sum(axis=1)
    dist_centrality = b.sum(axis=0) + b.sum(axis=1)
    
    heavy = np.argsort(-flow_centrality, kind="stable")
    central = np.argsort(dist_centrality, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _tabu_search(a, b, p_init, max_time, rng):
    """
    Tabu Search for QAP.
    """
    n = len(a)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    tabu_length = 15
    diversification_threshold = 100 # Iterations without improvement
    no_improve_count = 0
    current_iter = 0
    
    # Tabu matrix: tabu_until[i, j] = iteration until which (i,j) is tabu
    tabu_until = np.zeros((n, n), dtype=np.int64)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        best_move_i = -1
        best_move_j = -1
        best_delta = np.inf
        best_new_cost = current_cost
        
        # Scan all pairs to find the best non-tabu (or aspirated) move
        for i in range(n):
            for j in range(i + 1, n):
                # Check tabu status
                if tabu_until[i, j] > current_iter:
                    # It's tabu. Check aspiration: does it improve global best?
                    # We need to calculate delta to know new cost.
                    # To save time, we can skip if we are confident it's not better?
                    # But we don't know delta yet.
                    # For efficiency, we might skip tabu moves unless we are stuck.
                    # However, standard TS calculates delta for all, then filters.
                    # Or filters first if we have a heuristic.
                    # Let's calculate delta for all, it's the bottleneck anyway.
                    pass
                
                delta = _swap_delta(a, b, p, i, j)
                new_cost = current_cost + delta
                
                is_tabu = tabu_until[i, j] > current_iter
                
                if is_tabu:
                    # Aspiration criterion: accept if better than global best
                    if new_cost >= best_cost:
                        continue
                    # If it's tabu but improves global best, it's a candidate
                    # But we prefer non-tabu moves if they are close.
                    # Simple TS: if tabu, ignore unless aspiration.
                    # Here we allow it if aspiration holds.
                
                # If not tabu, or tabu with aspiration:
                if not is_tabu or new_cost < best_cost:
                    if delta < best_delta:
                        best_delta = delta
                        best_move_i = i
                        best_move_j = j
                        best_new_cost = new_cost
        
        if best_move_i == -1:
            # No valid move found (all tabu and not improving best)
            # Diversification: Random perturbation
            num_swaps = 5
            for _ in range(num_swaps):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            
            current_cost = _qap_cost(a, b, p)
            # Reset tabu list
            tabu_until[:] = 0
            current_iter = 0
            no_improve_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            continue
        
        # Apply best move
        p[best_move_i], p[best_move_j] = p[best_move_j], p[best_move_i]
        current_cost = best_new_cost
        
        # Update tabu list
        tabu_until[best_move_i, best_move_j] = current_iter + tabu_length
        tabu_until[best_move_j, best_move_i] = current_iter + tabu_length
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification if stuck
        if no_improve_count > diversification_threshold:
            num_swaps = 10
            for _ in range(num_swaps):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            
            current_cost = _qap_cost(a, b, p)
            tabu_until[:] = 0
            current_iter = 0
            no_improve_count = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        current_iter += 1
        
        # Time check inside loop is expensive, do it every few iterations
        if current_iter % 10 == 0:
            if time.time() - start_time > max_time:
                break

    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    max_total_time = 115.0
    
    rng = np.random.default_rng(42)
    
    best_p = None
    best_cost = np.inf
    
    # Run 1: Rank matching + Tabu Search
    p_init = _init_rank_matching(a, b)
    p_result, cost_result = _tabu_search(a, b, p_init, 50.0, rng)
    if cost_result < best_cost:
        best_cost = cost_result
        best_p = p_result
    
    # Run 2: Perturb best, Tabu Search
    if time.time() - start_time < max_total_time - 40.0:
        p_init = best_p.copy()
        n_swaps = n // 10
        for _ in range(n_swaps):
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            if i != j:
                p_init[i], p_init[j] = p_init[j], p_init[i]
        p_result, cost_result = _tabu_search(a, b, p_init, 35.0, rng)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result
            
    # Run 3: Different Init (Greedy-like or Random) + Tabu Search
    if time.time() - start_time < max_total_time - 30.0:
        # Use a random permutation or a different heuristic
        p_init = np.arange(n)
        rng.shuffle(p_init)
        p_result, cost_result = _tabu_search(a, b, p_init, 30.0, rng)
        if cost_result < best_cost:
            best_cost = cost_result
            best_p = p_result

    return best_p.tolist()
# EVOLVE-BLOCK-END
