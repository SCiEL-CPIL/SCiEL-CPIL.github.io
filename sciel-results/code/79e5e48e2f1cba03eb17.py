"""Tabu Search with long-term memory and Island Model for QAP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """Compute cost change when swapping positions i and j."""
    x = p[i]
    y = p[j]
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    val = np.dot(diff_a, diff_b[p])
    return val - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]

def _greedy_init(a, b, seed=42):
    """Greedy rank-matching initialization."""
    n = len(a)
    rng = random.Random(seed)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.copy()

def _random_init(n, seed=42):
    """Random permutation."""
    rng = random.Random(seed)
    p = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        p[i], p[j] = p[j], p[i]
    return p

def _steepest_polish(a_sym, b, p, max_iter=500):
    """
    Steepest Descent Local Search: repeatedly find the best swap and apply it
    until no improvement is found or max_iter is reached.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a_sym // 2, b, p) if np.allclose(a_sym, a_sym.T) else _qap_cost(a_sym, b, p)
    # Note: a_sym is a + a.T. If a is symmetric, a = a_sym/2. 
    # For safety in cost calculation during polish, we should use the original 'a' if available, 
    # but here we only have a_sym. The delta calculation uses a_sym.
    # The cost function _qap_cost takes 'a'. If we pass a_sym//2, it's correct for symmetric a.
    # Let's assume symmetry for tai256c.
    
    for _ in range(max_iter):
        best_delta = 0
        best_move = None
        # Sample a large subset or all moves? n=256 -> 32640 pairs. 
        # Full scan is feasible in NumPy if vectorized, but _swap_delta is scalar.
        # Let's sample a good number of random pairs to find a good move quickly.
        num_samples = 200
        for _ in range(num_samples):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            delta = _swap_delta(a_sym, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_move = (i, j)
        
        if best_move is None:
            break
        
        i, j = best_move
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
    return p, current_cost

def _tabu_search(a, b, p, time_limit, start_time, seed=42):
    """
    Tabu Search with:
    - Short-term memory (tabu list)
    - Long-term memory (frequency-based diversification)
    - Intensification via aspiration criteria
    """
    n = len(a)
    rng = random.Random(seed)
    a_sym = a + a.T  # Precompute symmetric flow matrix
    
    # Tabu parameters
    tabu_tenure = 30  # How many iterations a move stays tabu
    tabu_list = np.zeros((n, n), dtype=bool)  # tabu_list[i,j] = True if swap (i,j) is tabu
    move_history = []  # List of (iteration, i, j) for un-tabu-ing
    
    # Long-term memory: frequency of visits to each facility-location pair
    freq = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        freq[i, p[i]] += 1
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    best_iteration = 0
    no_improve = 0
    
    # Parameters for long-term memory effect
    freq_weight = 0.02
    freq_threshold = 200  # After this many iterations, start applying frequency penalty
    
    iteration = 0
    diversification_interval = 5000  # Every N iterations, add frequency penalty
    
    while time.time() - start_time < time_limit - 5:
        # Check if we need to un-tabu old moves
        while move_history and move_history[0][0] + tabu_tenure <= iteration:
            _, ti, tj = move_history.pop(0)
            tabu_list[ti, tj] = False
            tabu_list[tj, ti] = False
        
        # Generate candidate moves: sample a subset
        num_candidates = 100  # Increased from 50 for better exploration
        best_delta = np.inf
        best_move = None
        best_is_tabu = False
        best_aspiration = False
        
        for _ in range(num_candidates):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            
            delta = _swap_delta(a_sym, b, p, i, j)
            
            # Frequency penalty for diversification
            if iteration > freq_threshold:
                # Penalize moves to frequently visited locations
                freq_penalty = freq_weight * (freq[i, p[i]] + freq[j, p[j]])
                adjusted_delta = delta + freq_penalty
            else:
                adjusted_delta = delta
            
            is_tabu = tabu_list[i, j]
            
            # Aspiration criterion: accept tabu move if it improves best cost
            aspiration = is_tabu and (current_cost + delta < best_cost)
            
            # Choose best move
            if not is_tabu and delta < best_delta:
                best_delta = adjusted_delta
                best_move = (i, j)
                best_is_tabu = False
                best_aspiration = False
            elif is_tabu and aspiration and delta < best_delta:
                best_delta = adjusted_delta
                best_move = (i, j)
                best_is_tabu = True
                best_aspiration = True
        
        if best_move is None:
            # No valid move found, try a random move
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                i = (i + 1) % n
            best_move = (i, j)
            best_is_tabu = False
            best_aspiration = False
            best_delta = _swap_delta(a_sym, b, p, i, j)
        
        i, j = best_move
        x = p[i]
        y = p[j]
        
        # Perform the move
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update frequency
        freq[i, y] += 1
        freq[j, x] += 1
        freq[i, x] -= 1
        freq[j, y] -= 1
        
        # Update tabu list
        if not best_is_tabu or best_aspiration:
            tabu_list[i, j] = True
            tabu_list[j, i] = True
            move_history.append((iteration, i, j))
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            best_iteration = iteration
            no_improve = 0
        else:
            no_improve += 1
        
        iteration += 1
        
        # Diversification: if stuck for too long, restart with perturbation
        if no_improve > 2000:
            # Perturb current solution
            num_perturb = rng.randint(3, 8)
            for _ in range(num_perturb):
                pi = rng.randint(0, n - 1)
                pj = rng.randint(0, n - 1)
                if pi != pj:
                    p[pi], p[pj] = p[pj], p[pi]
            current_cost = _qap_cost(a, b, p)
            no_improve = 0
            # Clear tabu list
            tabu_list.fill(False)
            move_history.clear()
        
        # Safety check
        if time.time() - start_time > time_limit - 8:
            break
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    best_p = None
    best_cost = np.inf
    
    # Island Model: Run K independent trajectories
    # Each island uses a different initialization and random seed
    # This combines the structural diversity of the inspiration with the 
    # robust tabu search machinery of the current program.
    
    K = 4  # Number of islands
    time_per_island = time_limit / K
    
    for k in range(K):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 10:
            break
        
        # Different initialization strategies for each island
        if k == 0:
            p_init = _greedy_init(a, b, seed=42 + k)
        elif k == 1:
            p_init = _random_init(n, seed=123 + k)
        elif k == 2:
            # Hybrid: start from greedy, then random swaps
            p_init = _greedy_init(a, b, seed=42 + k)
            rng = random.Random(999 + k)
            for _ in range(n // 4):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
        else:
            # Another random perturbation of greedy
            p_init = _greedy_init(a, b, seed=42 + k)
            rng = random.Random(555 + k)
            for _ in range(n // 2):
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i != j:
                    p_init[i], p_init[j] = p_init[j], p_init[i]
        
        # Run tabu search for this island
        ts_time = min(remaining_time - 5, time_per_island)
        p_best, cost_best = _tabu_search(a, b, p_init.copy(), ts_time, start_time, seed=1000 + k)
        
        # Apply steepest polish to the result of this island
        if cost_best < best_cost:
            # Only polish if this island found a new best
            # This saves time for non-improving islands
            p_polished, cost_polished = _steepest_polish(a + a.T, b, p_best, max_iter=200)
            if cost_polished < cost_best:
                cost_best = cost_polished
                p_best = p_polished
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        best_p = _greedy_init(a, b, seed=0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
