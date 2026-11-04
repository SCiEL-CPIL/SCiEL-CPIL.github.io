"""
QAP Solver: Tabu Search with Aspiration and Strategic Oscillation.

Strategy:
1. Representation: Permutation p of facilities to locations.
2. Neighborhood: Swap of two facilities (i, j).
3. Move Evaluation: Vectorized O(n) calculation of the change in cost for a swap.
4. Search Mechanism: 
   - Select the best non-tabu move (aspiration if it improves the global best).
   - Maintain a tabu list of recently used facility indices (or pairs) to prevent cycling.
   - Strategic Oscillation: If stuck in a local minimum (no improving move for some time), 
     allow moves that worsen the cost slightly to escape, or restart with a perturbation.
5. Parallelism: Multiple independent Tabu Search instances with different seeds and 
   different tabu list lengths/parameters to explore diverse regions.
6. Initialization: Spectral initialization (eigenvector alignment) to provide a good starting point, 
   followed by a short local search (hill climbing) to polish.

Key difference from Simulated Annealing:
- Deterministic selection of the best move (steepest descent) rather than random moves with probabilistic acceptance.
- Explicit memory (tabu list) to enforce diversification and avoid cycles, rather than relying on temperature cooling.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
import os

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, rng):
    """
    Generate an initial permutation using spectral alignment.
    Aligns the principal eigenvectors of A and B.
    """
    n = len(a)
    # Compute eigenvectors
    # For symmetric matrices, eigenvectors are orthogonal.
    # We want to map high flow facilities to low distance locations?
    # Actually, we want to minimize sum a_ij b_pi(i) pi(j).
    # This is related to maximizing the inner product of the matrices if we flip signs?
    # Standard approach: sort facilities by eigenvalue components and locations by eigenvalue components.
    
    # Use SVD or Eigh. Eigh is faster for symmetric.
    try:
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
    except:
        return rng.permutation(n)
    
    # Take the first few eigenvectors to capture structure
    # Usually the largest magnitude eigenvalues correspond to the most significant structure.
    # Let's take the top k eigenvectors.
    k = min(4, n)
    
    # Sort eigenvalues by magnitude descending
    idx_a = np.argsort(np.abs(w_a))[::-1]
    idx_b = np.argsort(np.abs(w_b))[::-1]
    
    u_a = v_a[:, idx_a[:k]]
    u_b = v_b[:, idx_b[:k]]
    
    # Compute the soft assignment matrix X = U_a U_b^T
    # X[i, j] represents the "affinity" of facility i to location j.
    X = u_a @ u_b.T
    
    # Normalize X to be non-negative? Or just use the values.
    # We want to assign facility i to location p[i] that maximizes X[i, p[i]]?
    # Or minimizes?
    # Let's check the sign.
    # If A and B have positive correlation in their principal components, we want to align them.
    # The objective is sum a_ij b_pi(i) pi(j).
    # If we define X = U_a U_b^T, then trace(A X B X^T) is maximized when X aligns.
    # This is a heuristic. Let's just pick the best assignment from X.
    
    # To handle signs, we can flip the sign of columns of U_a or U_b to maximize the sum of X.
    # For each column j of U_b, flip sign if it makes the sum of X[:, j] negative?
    # Actually, simpler: just use the absolute values or raw values.
    
    # Let's use the Hungarian algorithm for exact assignment on the affinity matrix.
    # scipy.optimize.linear_sum_assignment minimizes cost.
    # So we want to minimize -X (maximize X).
    
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-X)
        p = np.zeros(n, dtype=int)
        p[row_ind] = col_ind
        return p
    except:
        return rng.permutation(n)

def _tabu_worker(a, b, seed, time_limit, start_time, tabu_len, aspiration_threshold):
    """
    Worker for Tabu Search.
    """
    n = len(a)
    rng = np.random.default_rng(seed)
    
    # Initialization
    p = _spectral_init(a, b, rng)
    
    # Polish with simple hill climbing (best improvement swap) for a short time
    # This ensures we start at a local minimum.
    # Precompute symmetric A
    a_sym = a + a.T
    diag_a = np.diag(a)
    diag_b = np.diag(b)
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    def get_delta(i, j):
        """Calculate delta for swapping facilities i and j in permutation p."""
        x = p[i]
        y = p[j]
        
        # Vectorized calculation
        vda = a_sym[i] - a_sym[j]
        vdb = b[y, p] - b[x, p]
        
        full_dot = np.dot(vda, vdb)
        term_i = vda[i] * vdb[i]
        term_j = vda[j] * vdb[j]
        sum_part = full_dot - term_i - term_j
        diag_part = (diag_a[i] - diag_a[j]) * (diag_b[y] - diag_b[x])
        delta = sum_part + diag_part
        
        return delta

    # Hill Climbing Initialization
    hc_start = time.time()
    hc_limit = 5.0 # 5 seconds for initial polish
    while time.time() - hc_start < hc_limit:
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample a subset of swaps to speed up initial polish? 
        # For n=256, O(n^2) is 65k swaps. Each O(n) is 256 ops. Total ~16M ops per pass.
        # This might be too slow for 5 seconds if we do many passes.
        # Let's do best-improvement on a random subset of 1000 swaps per pass.
        
        found_improvement = False
        for _ in range(1000):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            d = get_delta(i, j)
            if d < best_delta:
                best_delta = d
                best_i, best_j = i, j
        
        if best_i != -1 and best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            found_improvement = True
        else:
            break
            
    # Tabu Search Main Loop
    # Tabu list: store the last k moves (i, j). 
    # If (i, j) is in tabu list, it is forbidden unless aspiration criterion met.
    # Using a set of tuples is slow. 
    # Better: maintain an array tabu_until[i] = iteration when facility i becomes non-tabu.
    # But swapping i and j makes both i and j tabu? 
    # Standard QAP tabu: the move is defined by the pair of facilities swapped.
    # If we swap i and j, then the inverse move (swapping i and j again) is tabu.
    # So we can mark facilities i and j as tabu for T iterations.
    
    tabu_until = np.zeros(n, dtype=int)
    iteration = 0
    
    # Parameters
    max_iterations = 1000000 # Safety limit, time is the real limit
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        best_delta = 0
        best_i, best_j = -1, -1
        best_aspiration = False
        
        # Evaluate a set of candidate moves
        # To balance speed and quality, we can evaluate all swaps if fast enough, 
        # or a large random sample.
        # n=256, n^2/2 = 32640 swaps.
        # Vectorized evaluation of all swaps is difficult without O(n^3) memory or time.
        # Let's use a random sample of 2000 moves per iteration.
        
        num_candidates = 2000
        for _ in range(num_candidates):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
            
            d = get_delta(i, j)
            
            # Check tabu
            is_tabu = (iteration < tabu_until[i]) or (iteration < tabu_until[j])
            
            # Aspiration: accept tabu move if it improves global best
            aspiration = (d < best_cost)
            
            if is_tabu and not aspiration:
                continue
            
            if d < best_delta:
                best_delta = d
                best_i, best_j = i, j
                best_aspiration = aspiration
        
        if best_i == -1:
            # No valid move found (all tabu or no improvement and no aspiration)
            # If no aspiration and no improvement, we are stuck.
            # Break or perturb?
            # Let's perturb: pick a random move.
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                i = (i + 1) % n
            d = get_delta(i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += d
            tabu_until[i] = iteration + tabu_len
            tabu_until[j] = iteration + tabu_len
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            # Reheat/Restart logic: if stuck for too long, restart from best_p with perturbation
            # For now, just continue.
        else:
            # Apply best move
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            tabu_until[best_i] = iteration + tabu_len
            tabu_until[best_j] = iteration + tabu_len
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Diversification: Every 1000 iterations, if no improvement in last 1000, perturb
        # We don't track no-improvement count easily here without extra state.
        # Let's just rely on the random sampling and tabu list.
        
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    effective_limit = time_limit - safety_margin
    
    # Number of workers
    n_workers = min(mp.cpu_count(), 8)
    
    if n_workers <= 1:
        p_best, cost_best = _tabu_worker(a, b, 42, effective_limit, start_time, tabu_len=20, aspiration_threshold=0)
        return p_best.tolist()
    
    # Parallel execution with different tabu lengths
    # Diversify by using different tabu lengths
    tabu_lengths = [10, 20, 30, 50, 10, 20, 30, 50][:n_workers]
    seeds = [i * 1000 + 42 for i in range(n_workers)]
    
    args = [(a, b, seeds[i], effective_limit, start_time, tabu_lengths[i], 0) for i in range(n_workers)]
    
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(_tabu_worker, args)
    except Exception as e:
        # Fallback to single thread
        p_best, cost_best = _tabu_worker(a, b, 42, effective_limit, start_time, tabu_len=20, aspiration_threshold=0)
        return p_best.tolist()
    
    global_best_p = None
    global_best_cost = np.inf
    
    for p, cost in results:
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_p = p
            
    if global_best_p is None:
        return np.random.permutation(n).tolist()
        
    return global_best_p.tolist()

# EVOLVE-BLOCK-END
