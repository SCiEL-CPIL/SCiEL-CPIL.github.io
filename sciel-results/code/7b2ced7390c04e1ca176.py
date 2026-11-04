"""
Parallel Iterated Tabu Search (P-ITS) for QAPLIB wil100.

Strategy:
1. Initialize multiple diverse permutations using a linear surrogate (Hungarian) 
   with randomized tie-breaking or slight perturbations.
2. Run independent Tabu Search threads in parallel (conceptually, sequentially in 
   Python but structured to maximize diversity).
3. Each thread uses:
   - Short-term tabu list (recency-based) to prevent immediate reversals.
   - Aspiration criterion to override tabu status if a move improves the best known solution.
   - Dynamic neighborhood: start with small random swaps, expand to larger perturbations 
     (k-swaps) if stuck at a local optimum for a certain number of iterations.
4. Vectorized delta evaluation for O(n) cost change calculation.
5. Return the best permutation found across all threads within the time limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _generate_initial_permutation(a, b, rng):
    """
    Generate a diverse initial permutation using Hungarian algorithm on a 
    perturbed surrogate cost matrix.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Base surrogate
    C = -np.outer(a_row_sums, b_row_sums)
    
    # Add random noise to break ties and create diversity
    noise_scale = np.std(C) * 0.1
    C_perturbed = C + rng.normal(0, noise_scale, C.shape)
    
    row_ind, col_ind = linear_sum_assignment(C_perturbed)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _tabu_search_thread(a, b, p_init, best_cost_ref, best_p_ref, time_limit, rng, thread_id):
    """
    Run a single Tabu Search thread.
    Updates best_cost_ref and best_p_ref if a better solution is found.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    
    # Tabu list: store (i, j, expiry_iter)
    tabu_list = []
    tabu_tenure = 15  # Number of iterations a move is forbidden
    
    # Track stagnation
    stagnant_count = 0
    best_thread_cost = current_cost
    best_thread_p = p.copy()
    
    start_time = time.time()
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Determine neighborhood size
        # If stagnant, expand neighborhood significantly (perturbation)
        if stagnant_count > 20:
            # Large perturbation: k random swaps
            k_swaps = rng.randint(5, 15)
            for _ in range(k_swaps):
                i = rng.randint(n)
                j = rng.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = _qap_cost(a, b, p)
            stagnant_count = 0
            tabu_list.clear()  # Clear tabu after big jump
            continue
        
        # Normal tabu search step
        best_delta = np.inf
        best_move = None
        best_aspiration = False
        
        # Sample a subset of moves to keep it fast (e.g., 100 random pairs)
        # Or scan all if n is small, but for n=100, sampling is faster
        num_samples = 100
        for _ in range(num_samples):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            delta = _swap_delta(a, b, p, i, j)
            
            # Check tabu status
            is_tabu = False
            for (ti, tj, expiry) in tabu_list:
                if (ti == i and tj == j) and expiry > iter_count:
                    is_tabu = True
                    break
            
            # Aspiration criterion: accept if it improves global best
            aspiration = (not is_tabu) or (current_cost + delta < best_cost_ref[0])
            
            if aspiration:
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
                    best_aspiration = True
        
        if best_move is None:
            # No valid move found, force a random swap
            i = rng.randint(n)
            j = rng.randint(n)
            while i == j:
                j = rng.randint(n)
            best_move = (i, j)
            best_delta = _swap_delta(a, b, p, i, j)
        
        i, j = best_move
        
        # Apply move
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update tabu list
        tabu_list.append((i, j, iter_count + tabu_tenure))
        # Remove expired tabu entries
        tabu_list = [(ti, tj, exp) for (ti, tj, exp) in tabu_list if exp > iter_count]
        
        # Update bests
        if current_cost < best_thread_cost:
            best_thread_cost = current_cost
            best_thread_p = p.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
            
        if current_cost < best_cost_ref[0]:
            best_cost_ref[0] = current_cost
            best_p_ref[0] = p.copy()
    
    return best_thread_cost, best_thread_p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0  # Safety margin before 120s limit
    start_time = time.time()
    
    # Use a fixed seed for reproducibility of the overall process, 
    # but different seeds for each thread to ensure diversity
    master_rng = np.random.RandomState(42)
    
    # Number of parallel threads (sequential execution in Python, but distinct trajectories)
    num_threads = 4
    
    # Shared best solution
    best_cost = [np.inf]
    best_p = [np.zeros(n, dtype=np.int64)]
    
    # Generate initial permutations for each thread
    initial_perts = []
    for t in range(num_threads):
        thread_rng = np.random.RandomState(master_rng.randint(1000000))
        p_init = _generate_initial_permutation(a, b, thread_rng)
        initial_perts.append((p_init, thread_rng))
    
    # Distribute time among threads
    # To maximize diversity, we run them in a round-robin fashion or just sequentially 
    # with equal time budgets. Since Python is single-threaded for CPU-bound NumPy,
    # we just run them sequentially but with independent state.
    
    time_per_thread = total_deadline / num_threads
    
    for t in range(num_threads):
        p_init, thread_rng = initial_perts[t]
        
        # Check if we already have a very good solution, maybe reduce time for later threads?
        # For simplicity, keep equal time.
        
        remaining_time = total_deadline - (time.time() - start_time)
        if remaining_time <= 0:
            break
            
        # Cap thread time to not overshoot
        actual_thread_time = min(time_per_thread, remaining_time)
        
        _tabu_search_thread(a, b, p_init, best_cost, best_p, actual_thread_time, thread_rng, t)
        
        # Check if we are close to the deadline
        if time.time() - start_time > total_deadline - 1.0:
            break

    # Final polish: if we have time left, do a quick SA or local search on the best found
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 2.0:
        # Quick local search on best_p
        p_ls = best_p[0].copy()
        cost_ls = best_cost[0]
        
        # Simple first-improvement local search
        improved = True
        while improved and (time.time() - start_time) < total_deadline - 1.0:
            improved = False
            for i in range(n):
                if (time.time() - start_time) > total_deadline - 1.0:
                    break
                for j in range(i + 1, n):
                    delta = _swap_delta(a, b, p_ls, i, j)
                    if delta < 0:
                        p_ls[i], p_ls[j] = p_ls[j], p_ls[i]
                        cost_ls += delta
                        improved = True
                        if cost_ls < best_cost[0]:
                            best_cost[0] = cost_ls
                            best_p[0] = p_ls.copy()
        
    return best_p[0].tolist()
# EVOLVE-BLOCK-END
