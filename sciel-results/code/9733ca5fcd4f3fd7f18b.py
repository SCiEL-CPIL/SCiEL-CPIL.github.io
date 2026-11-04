"""
QAP Solver: Parallel Multi-Start Simulated Annealing with First-Improvement Neighborhood.

Strategy:
1. Parallelism: Utilize multiprocessing to run independent Simulated Annealing (SA) instances concurrently.
2. Neighborhood: Use a first-improvement search strategy. Instead of evaluating all O(n^2) swaps to find the best (steepest descent), we evaluate swaps in a random or pseudo-random order and accept the first one that improves the solution (or satisfies SA criteria). This is significantly faster per iteration than best-improvement.
3. Delta Calculation: Precompute matrices to allow O(n) vectorized calculation of the cost delta for swapping any two facilities.
4. Diversification: Each parallel process uses a different random seed and initialization (random or greedy) to explore distinct regions of the search space.
5. Time Management: Each process runs until a shared deadline (with safety margin) is reached, then returns its best found solution. The main process aggregates results and returns the global best.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import multiprocessing as mp
import os

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sa_worker(a, b, seed, time_limit, start_time):
    """
    Worker function for one Simulated Annealing instance.
    Returns (best_p, best_cost).
    """
    n = len(a)
    rng = np.random.default_rng(seed)
    
    # Initialization: Random permutation
    p = rng.permutation(n)
    
    # Precompute symmetric A for delta calc
    # Note: QAP cost is sum a_ij b_pi(i) pi(j).
    # If we swap i and j, the change involves a[i,k]+a[k,i] terms.
    a_sym = a + a.T
    
    # Precompute B rows? No, we need b[p[i], p[j]] etc.
    # We will maintain a permuted B matrix B_perm = b[np.ix_(p, p)]?
    # Actually, maintaining B_perm is expensive to update (O(n^2) copy).
    # Better to keep p and access b[p[i], p[j]].
    
    # Fast Delta Calculation for swapping facilities i and j:
    # Let x = p[i], y = p[j].
    # Delta = (a[i,i]-a[j,j]) * (b[y,y]-b[x,x]) + 
    #         sum_{k != i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[y, p[k]] - b[x, p[k]])
    # Let A_diff_k = a_sym[i, k] - a_sym[j, k]
    # Let B_diff_k = b[y, p[k]] - b[x, p[k]]
    # Sum = dot(A_diff, B_diff)
    # But we must exclude k=i and k=j from the sum.
    # A_diff[i] = a_sym[i,i] - a_sym[j,i] = 2a[i,i] - a[i,j] - a[j,i]
    # B_diff[i] = b[y, p[i]] - b[x, p[i]] = b[y, x] - b[x, x]
    # A_diff[j] = a_sym[i,j] - a_sym[j,j] = a[i,j] + a[j,i] - 2a[j,j]
    # B_diff[j] = b[y, p[j]] - b[x, p[j]] = b[y, y] - b[x, y]
    # 
    # Actually, it's easier to just compute the full dot product and subtract the i and j terms.
    # Full Dot = sum_{k=0}^{n-1} (a_sym[i,k] - a_sym[j,k]) * (b[y, p[k]] - b[x, p[k]])
    # Term i: (a_sym[i,i] - a_sym[j,i]) * (b[y, p[i]] - b[x, p[i]])
    # Term j: (a_sym[i,j] - a_sym[j,j]) * (b[y, p[j]] - b[x, p[j]])
    # 
    # Wait, b[y, p[i]] where p[i]=x is b[y,x].
    # b[x, p[i]] where p[i]=x is b[x,x].
    # So Term i = (a_sym[i,i] - a_sym[j,i]) * (b[y,x] - b[x,x])
    # 
    # Term j: p[j]=y.
    # b[y, p[j]] = b[y,y].
    # b[x, p[j]] = b[x,y].
    # So Term j = (a_sym[i,j] - a_sym[j,j]) * (b[y,y] - b[x,y])
    
    # Let's verify the diagonal terms in the original sum.
    # Original sum excludes i, j.
    # So Delta = (Full Dot - Term i - Term j) + (a[i,i]-a[j,j])*(b[y,y]-b[x,x])?
    # No, the formula derived earlier was:
    # Delta = (a[i,i]-a[j,j])(b[y,y]-b[x,x]) + Sum_{k!=i,j} ...
    # The "Sum" part is exactly (Full Dot - Term i - Term j).
    
    # So:
    # 1. Compute vec_diff_a = a_sym[i, :] - a_sym[j, :]
    # 2. Compute vec_diff_b = b[y, p] - b[x, p]  (Vector of length n)
    # 3. full_dot = np.dot(vec_diff_a, vec_diff_b)
    # 4. term_i = vec_diff_a[i] * vec_diff_b[i]
    # 5. term_j = vec_diff_a[j] * vec_diff_b[j]
    # 6. sum_part = full_dot - term_i - term_j
    # 7. diag_part = (a[i,i] - a[j,j]) * (b[y,y] - b[x,x])
    # 8. delta = sum_part + diag_part
    
    # This is O(n) per swap evaluation.
    # First-improvement: iterate through random pairs or sequential pairs?
    # Random pairs is better for diversification.
    # To make it fast, we can generate a list of random pairs and check them.
    # Or just pick random i, j.
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # SA Parameters
    # Tail256c is large.
    # T0 = 1000, T_end = 1, alpha = 0.99?
    # We don't know the scale of costs.
    # Cost is ~44M.
    # Deltas can be large.
    # Let's estimate typical delta magnitude.
    # Random walk.
    # Let's set T0 = 10000, cool to 1.
    
    T = 10000.0
    T_end = 1.0
    # We want to finish in time_limit.
    # We don't know how many iterations we can do.
    # Let's just loop until time runs out.
    # Cooling: T = T * alpha + T_end * (1-alpha)?
    # Or simply T = max(T_end, T * 0.999)
    
    alpha = 0.999
    
    # Precompute diagonal of A and B for speed
    diag_a = np.diag(a)
    diag_b = np.diag(b)
    
    # To speed up vec_diff_b calculation:
    # b[y, p] - b[x, p]
    # We can't precompute this easily for all y, x because p changes.
    # But we can just do the subtraction. It's O(n).
    
    # Optimization:
    # Instead of generating random i, j every time, we can maintain a "candidate" list?
    # No, random is fine.
    
    # First-improvement logic:
    # In each "step" (which corresponds to one cooling step in standard SA),
    # we might try multiple swaps?
    # Standard SA: pick one random move, accept/reject.
    # First-improvement SA: pick a neighborhood (e.g., all swaps), find first improving one.
    # If found, accept and reset temperature? Or continue cooling?
    # If not found, cool down.
    
    # Let's do:
    # While time remains:
    #   Try to find an improving move by sampling up to K random pairs.
    #   If found, accept it.
    #   If not found, accept a random move with prob exp(-delta/T) (to escape), else cool T.
    
    K = 50  # Number of random pairs to check for improvement per step
    
    while time.time() - start_time < time_limit:
        # Try to find first improvement
        found_improvement = False
        delta_best = 0
        i_best = -1
        j_best = -1
        
        for _ in range(K):
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue
                
            x = p[i]
            y = p[j]
            
            # Calculate delta
            # vec_diff_a = a_sym[i] - a_sym[j]
            # vec_diff_b = b[y, p] - b[x, p]
            
            # Optimization: 
            # a_sym[i] and a_sym[j] are rows.
            # b[y, p] and b[x, p] are fancy indexing.
            
            # Let's try to avoid allocating large temp arrays if possible?
            # np.dot does the allocation internally.
            
            vda = a_sym[i] - a_sym[j]
            vdb = b[y, p] - b[x, p]
            
            full_dot = np.dot(vda, vdb)
            term_i = vda[i] * vdb[i]
            term_j = vda[j] * vdb[j]
            sum_part = full_dot - term_i - term_j
            diag_part = (diag_a[i] - diag_a[j]) * (diag_b[y] - diag_b[x])
            delta = sum_part + diag_part
            
            if delta < 0:
                # Improving move
                found_improvement = True
                delta_best = delta
                i_best = i
                j_best = j
                break
        
        if found_improvement:
            # Accept move
            p[i_best], p[j_best] = p[j_best], p[i_best]
            current_cost += delta_best
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            # Do not cool? Or cool slightly?
            # Standard first-improvement SA often keeps T constant or cools slowly if improving.
            # Let's cool slowly to allow exploration.
            T = max(T_end, T * 0.9999)
        else:
            # No improvement found in K samples.
            # Try a random move with SA acceptance
            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                # Just cool
                T = max(T_end, T * alpha)
                continue
                
            x = p[i]
            y = p[j]
            vda = a_sym[i] - a_sym[j]
            vdb = b[y, p] - b[x, p]
            full_dot = np.dot(vda, vdb)
            term_i = vda[i] * vdb[i]
            term_j = vda[j] * vdb[j]
            sum_part = full_dot - term_i - term_j
            diag_part = (diag_a[i] - diag_a[j]) * (diag_b[y] - diag_b[x])
            delta = sum_part + diag_part
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            # Cool down
            T = max(T_end, T * alpha)
            
            # If T is very low, we might get stuck. 
            # Reheat?
            if T <= T_end * 1.01:
                T = 1000.0 # Reheat to escape local minima
                
    return best_p, best_cost

def _init_worker(a, b):
    """
    Initialize worker globals to avoid passing large arrays via pickle if using Pool?
    Actually, passing arrays to Pool workers is done once.
    But multiprocessing on Windows/macOS uses spawn, so globals are not shared.
    On Linux, fork is default, so globals are shared.
    To be safe and portable, we pass a, b to the worker function.
    However, pickling 256x256 int64 arrays is fast (~100KB).
    """
    pass

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    effective_limit = time_limit - safety_margin
    
    # Number of workers
    # Let's use min(cpu_count, 8) to avoid overhead
    n_workers = min(mp.cpu_count(), 8)
    
    # If n_workers is 1, just run inline to avoid overhead
    if n_workers <= 1:
        # Run single instance
        p_best, cost_best = _sa_worker(a, b, 42, effective_limit, start_time)
        return p_best.tolist()
    
    # Parallel execution
    # We need to pass a, b to each worker.
    # Using Pool.map is convenient.
    
    seeds = [i * 1000 + 42 for i in range(n_workers)]
    # Each worker gets the same time limit (relative to start)
    # But _sa_worker takes start_time.
    # We need to pass the global start_time to all workers.
    
    # Note: time.time() in worker is local wall clock, which is synced.
    
    args = [(a, b, seed, effective_limit, start_time) for seed in seeds]
    
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(_sa_worker, args)
    except Exception as e:
        # Fallback to single thread if multiprocessing fails
        print(f"Multiprocessing failed: {e}")
        p_best, cost_best = _sa_worker(a, b, 42, effective_limit, start_time)
        return p_best.tolist()
    
    # Find global best
    global_best_p = None
    global_best_cost = np.inf
    
    for p, cost in results:
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_p = p
            
    if global_best_p is None:
        # Should not happen
        return np.random.permutation(n).tolist()
        
    return global_best_p.tolist()

# EVOLVE-BLOCK-END
