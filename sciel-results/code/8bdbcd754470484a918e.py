import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _compute_delta(a, b, p, i, j):
    """
    Compute the cost change for swapping facilities at positions i and j.
    a, b: symmetric matrices
    p: current permutation
    i, j: indices to swap
    """
    n = len(p)
    u = p[i]
    v = p[j]
    
    # Precompute differences
    # delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, k] - b[u, k]) 
    #         + (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[v, v] - b[u, u])
    
    # Vectorized sum over k
    # mask = np.ones(n, dtype=bool)
    # mask[i] = False
    # mask[j] = False
    # k_indices = np.where(mask)[0]
    
    # Instead of creating mask every time, we can sum all and subtract i, j terms
    # Sum_all = sum_{k=0}^{n-1} (a[i, k] - a[j, k]) * (b[v, k] - b[u, k])
    
    diff_a = a[i, :] - a[j, :]
    diff_b = b[v, :] - b[u, :]
    
    sum_all = np.dot(diff_a, diff_b)
    
    # Subtract terms for k=i and k=j
    # Term k=i: (a[i, i] - a[j, i]) * (b[v, i] - b[u, i])
    # Note: b[v, i] is B[v, p[i]] = B[v, u]. b[u, i] is B[u, p[i]] = B[u, u].
    # So term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
    
    # Term k=j: (a[i, j] - a[j, j]) * (b[v, j] - b[u, j])
    # b[v, j] is B[v, p[j]] = B[v, v]. b[u, j] is B[u, p[j]] = B[u, v].
    # So term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
    
    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
    
    sum_k_neq = sum_all - term_i - term_j
    
    # Self terms
    self_term = (a[i, i] + a[j, j] - a[i, j] - a[j, i]) * (b[v, v] - b[u, u])
    
    delta = 2 * sum_k_neq + self_term
    
    return int(delta)

def _tabu_search(a, b, p_init, cost_init, time_limit):
    """
    Run Tabu Search for a given time limit.
    """
    n = len(p)
    p = p_init.copy()
    cost = cost_init
    best_p = p.copy()
    best_cost = cost
    
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    tabu_size = 100
    current_iter = 0
    no_improve_count = 0
    perturb_interval = 5000
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        current_iter += 1
        
        # Find best non-tabu move
        best_delta = 0
        best_i = -1
        best_j = -1
        best_is_aspiration = False
        
        # Sample a subset of moves to save time? No, n=100, n^2/2 = 5000 moves.
        # 5000 moves * O(n) per move = 500,000 ops. ~0.005s per iteration.
        # 200 iterations/s. 150s -> 30,000 iterations.
        # This is feasible.
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check tabu
                if (i, j) in tabu_list and tabu_list[(i, j)] > current_iter:
                    # Check aspiration
                    # We need the cost to check aspiration. 
                    # To avoid computing delta for tabu moves that don't meet aspiration,
                    # we can skip them unless we track potential. 
                    # Simpler: just skip tabu moves. Aspiration is usually handled by checking if the move leads to a new best.
                    # But we don't know the delta without computing it.
                    # Let's compute delta for all, but only accept if non-tabu or aspiration.
                    pass
                
                delta = _compute_delta(a, b, p, i, j)
                
                is_tabu = (i, j) in tabu_list and tabu_list[(i, j)] > current_iter
                
                if is_tabu:
                    # Aspiration: if this move improves the best known cost
                    # We don't know the new cost without applying, but new_cost = cost + delta
                    # If cost + delta < best_cost, it's aspiration.
                    if cost + delta < best_cost:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        best_is_aspiration = True
                else:
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
                        best_is_aspiration = False
        
        if best_i == -1:
            # No improving moves found (and no aspiration)
            # This shouldn't happen if we allow any move, but we are looking for best delta.
            # If all deltas are positive, we are at a local minimum.
            # We should still pick the least bad move to escape? 
            # Standard Tabu Search picks the best move regardless of sign, unless it's tabu.
            # My loop above only updates best_delta if delta < best_delta (initialized to 0).
            # So if all deltas > 0, best_delta remains 0, best_i remains -1.
            # We need to allow worsening moves.
            
            # Re-evaluate: Find the absolute best move (most negative delta), ignoring tabu status for selection, 
            # but only reject if tabu AND not aspiration.
            # Actually, standard TS: Select best move from N(p) \ T, or from T if aspiration.
            # If N(p) \ T is empty, pick from T.
            # Let's restart the selection logic to be robust.
            
            best_delta = np.inf
            best_i = -1
            best_j = -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    delta = _compute_delta(a, b, p, i, j)
                    is_tabu = (i, j) in tabu_list and tabu_list[(i, j)] > current_iter
                    
                    if not is_tabu:
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                    else:
                        # Aspiration
                        if cost + delta < best_cost:
                            if delta < best_delta:
                                best_delta = delta
                                best_i = i
                                best_j = j
            
            if best_i == -1:
                # Should not happen, but break if stuck
                break

        # Perform the move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
        
        # Update tabu list
        tabu_list[(best_i, best_j)] = current_iter + tabu_size
        
        # Update best
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification
        if no_improve_count > perturb_interval:
            # Random perturbation
            num_perturb = 10
            for _ in range(num_perturb):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    delta = _compute_delta(a, b, p, i, j)
                    p[i], p[j] = p[j], p[i]
                    cost += delta
            # Reset tabu list
            tabu_list.clear()
            no_improve_count = 0
            
            # If perturbation improved best, update
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()

    return best_p, best_cost

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic based on row sums.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending centrality (or descending, depending on definition)
    # Usually, high flow facilities go to low distance (central) locations.
    # b_row_sums represents total distance to all other locations.
    # Low sum = central.
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin
    start_time = time.time()
    
    # Phase 1: Initialization
    p_greedy = _greedy_initial(a, b)
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = cost_greedy if cost_greedy < cost_random else cost_random
    
    # Phase 2: Tabu Search
    # Run for the majority of the time
    ts_time = total_deadline - 10.0  # Reserve 10s for final polish
    
    p_ts, cost_ts = _tabu_search(a, b, best_p, best_cost, ts_time)
    
    if cost_ts < best_cost:
        best_cost = cost_ts
        best_p = p_ts.copy()
        
    # Phase 3: Final Polish with Local Search (Steepest Descent)
    elapsed = time.time() - start_time
    polish_time = min(5.0, max(0.0, total_deadline - elapsed - 2.0))
    
    if polish_time > 0.5:
        # Simple local search to remove any last small improvements
        p_ls = best_p.copy()
        cost_ls = best_cost
        improved = True
        polish_start = time.time()
        
        while improved and (time.time() - polish_start) < polish_time:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    delta = _compute_delta(a, b, p_ls, i, j)
                    if delta < 0:
                        p_ls[i], p_ls[j] = p_ls[j], p_ls[i]
                        cost_ls += delta
                        improved = True
                        if cost_ls < best_cost:
                            best_cost = cost_ls
                            best_p = p_ls.copy()
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

    return best_p.tolist()
