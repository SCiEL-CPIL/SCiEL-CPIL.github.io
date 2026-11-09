"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_completion_matrix(perm, pt):
    """
    Computes the completion time matrix C for a given permutation.
    C[i, m] is the completion time of job perm[i] on machine m.
    """
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return np.zeros((0, m))
    
    # Initialize completion matrix
    C = np.zeros((n, m))
    
    # First job
    C[0, 0] = pt[perm[0], 0]
    if m > 1:
        C[0, 1:] = np.cumsum(pt[perm[0], 1:])
    
    # Subsequent jobs
    for i in range(1, n):
        job = perm[i]
        # C[i, 0] depends only on C[i-1, 0]
        C[i, 0] = C[i-1, 0] + pt[job, 0]
        if m > 1:
            # C[i, k] = max(C[i-1, k], C[i, k-1]) + pt[job, k]
            # We can vectorize the max part
            # C[i, 1:] = max(C[i-1, 1:], C[i, :-1]) + pt[job, 1:]
            C[i, 1:] = np.maximum(C[i-1, 1:], C[i, :-1]) + pt[job, 1:]
            
    return C

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Initial Solution: NEH Heuristic
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # NEH Insertion
    current_perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        # Try all positions
        # Optimization: We can just insert and compute, but for NEH speed, 
        # we can approximate or just do a quick check. 
        # Given n is likely small enough, we do full check but it's O(n^2*m).
        # To be safe on time, we limit NEH steps if n is huge, but usually n<100.
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            # Quick makespan calc for NEH
            # Since this is just initialization, we can afford O(n*m) per step
            # Or better, just use a simple heuristic if n is very large.
            # Let's stick to standard NEH for quality.
            # To speed up NEH, we can compute makespan efficiently.
            # But let's just compute it.
            mk = 0
            # Simple loop for NEH makespan
            c = 0
            for k in range(m):
                c += processing_times[job, k]
            # This is wrong, need flow shop logic.
            # Let's use the matrix function but it's overkill for NEH loop.
            # Let's just use a fast inline calculation.
            
            # Actually, let's just run NEH once. It's O(n^2 * m).
            # If n=100, 10000 * 10 = 100k ops. Very fast.
            
            # To avoid code duplication, I'll use a helper for makespan
            # but for NEH I will just use the matrix function at the end of NEH
            # and use a simpler greedy logic during NEH?
            # No, standard NEH requires evaluating makespan at each insertion.
            
            # Let's implement a fast makespan for the NEH phase.
            # C_m is completion on machine m for the last job.
            # This is tricky to do inline without a matrix.
            
            # Let's just use the _compute_completion_matrix for the final NEH perm
            # and for the NEH construction, we will use a slightly slower but 
            # acceptable O(n*m) loop for each insertion candidate.
            
            # Optimization: Since we are in a tight loop, let's just 
            # pick the position that minimizes the sum of processing times 
            # in the "critical path" approximation? No, let's do proper NEH.
            
            # I will implement a fast makespan function specifically for this.
            
            # For now, to keep code clean, I'll use the matrix function 
            # but only if n is small. If n is large, I'll use a simpler init.
            
            # Let's assume n is small enough for O(n^2 * m) NEH.
            pass
            
        # Re-evaluating NEH strategy:
        # The standard NEH is O(n^2 * m).
        # I will implement a fast _makespan that takes a list and pt.
        
        # Let's refine the loop above.
        
        # I'll restart the NEH logic properly below.
        pass
        
    # --- Refined NEH ---
    current_perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_mk = float('inf')
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            # Fast makespan for NEH
            # We can compute this in O(m * len(new_perm))
            # But doing this n^2 times is O(n^3 * m).
            # If n=50, 125k * m. Fine.
            # If n=200, 8M * m. Might be slow in Python.
            # Let's use a vectorized makespan for NEH too?
            # No, let's just use the matrix function for the final NEH result
            # and use a heuristic for the insertion position if n is large.
            
            # Actually, let's just do the insertion and compute makespan 
            # using the matrix function. It's O(n*m) per call.
            # Total NEH cost: Sum_{k=1}^n k * O(n*m) = O(n^3 * m).
            # This might be too slow for large n.
            
            # Alternative: Use a simpler init if n > 100.
            if n > 100:
                # Just insert at end or random if NEH is too slow?
                # No, let's optimize the makespan call.
                pass
            
            # Let's implement a very fast makespan for the NEH loop.
            # C[i] = max(C[i-1], C_prev) + p[i]
            # We can store C as a numpy array of length m.
            
            # Let's just use the matrix function for simplicity and correctness 
            # and hope n is small. The task says "local synthetic flow-shop instances".
            # Usually these are n=10-50.
            
            C = _compute_completion_matrix(new_perm, processing_times)
            mk = C[-1, -1]
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    # 2. Local Search with Delta Evaluation
    # We will use 2-Opt (swap) and Insertion moves.
    # We maintain the completion matrix C.
    # Delta calculation for swap (i, j):
    # This is complex. 
    # Alternative: Use "Fast Re-evaluation" by only recalculating the affected part?
    # In flow shop, changing position i affects all subsequent jobs on all machines.
    # So delta is O((n-i) * m).
    # If we pick random i, j, average cost is O(n*m).
    # Same as full re-eval.
    
    # To beat the current program, we need a better search method.
    # The current program uses SA with random swaps.
    # A local search that accepts ONLY improvements (Hill Climbing) is faster 
    # if we can evaluate neighbors quickly.
    
    # Strategy: 
    # 1. Compute C for current_perm.
    # 2. Generate all possible swaps (i, j) or insertions.
    # 3. For each, estimate the delta.
    # 4. If delta < 0, apply it and update C incrementally?
    #    Updating C incrementally for a swap is hard because the "prefix" changes.
    
    # Let's use a different approach: **Iterated Local Search (ILS)** with a 
    # fast perturbation and a very fast local search using a 
    # **Vectorized Neighbor Evaluation**.
    
    # Actually, the most effective improvement for flow shop is often 
    # **NEH + 2-Opt Local Search**.
    # The bottleneck is the makespan calculation.
    
    # Let's implement a Local Search that tries all O(n^2) swaps.
    # For each swap, we can compute the makespan in O(n*m).
    # Total cost per iteration: O(n^3 * m).
    # If n=50, 125k * 10 = 1.25M ops. Very fast in NumPy?
    # Python loop over n^2 swaps is slow.
    
    # Better: **Vectorized Swap Evaluation**.
    # We can't easily vectorize the *logic* of the flow shop makespan 
    # for arbitrary swaps without pre-computing prefix/suffix sums, 
    # which is possible but complex.
    
    # Let's stick to a robust, fast Python implementation of 
    # **2-Opt Local Search** with **early termination** and 
    # **multiple restarts** if time permits.
    
    # To make it faster than the current SA:
    # 1. Use a better initial solution (NEH is already there).
    # 2. Use Hill Climbing (accept only improvements) which converges faster 
    #    to a local optimum than SA which wanders.
    # 3. Use a more effective move: **Insertion** (remove job i, insert at j).
    #    Insertion moves are often better than swaps in permutation problems.
    
    # Let's implement **First-Improvement Insertion Local Search**.
    
    current_perm = np.array(current_perm)
    current_C = _compute_completion_matrix(current_perm, processing_times)
    current_mk = current_C[-1, -1]
    
    best_perm = current_perm.copy()
    best_mk = current_mk
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_jobs = len(current_perm)
        
        # Try all insertion moves
        # Move job i to position j (i != j)
        # To avoid O(n^3), we can try random insertions or all insertions.
        # All insertions is O(n^2) moves. Each O(n*m) eval. Total O(n^3*m).
        # For n=50, this is ~1.2M ops per pass. 
        # Python loop overhead might make this slow.
        
        # Let's try **Random Insertion Local Search** with many iterations.
        # Or **Vectorized Evaluation of a Batch of Moves**.
        
        # Let's try a batch of 100 random insertion moves per step.
        
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Pick a random job to move
            i = random.randint(0, n_jobs - 1)
            job = current_perm[i]
            
            # Pick a random destination (not i)
            j = random.randint(0, n_jobs - 1)
            if i == j:
                continue
                
            # Create new permutation
            # Remove job at i
            # Insert at j (adjust index if i < j)
            
            # Efficient perm manipulation:
            # new_perm = np.concatenate((current_perm[:i], current_perm[i+1:]))
            # if i < j: j -= 1
            # new_perm = np.insert(new_perm, j, job)
            
            # This is O(n).
            
            new_perm = np.delete(current_perm, i)
            if i < j:
                j -= 1
            new_perm = np.insert(new_perm, j, job)
            
            # Evaluate
            new_C = _compute_completion_matrix(new_perm, processing_times)
            new_mk = new_C[-1, -1]
            
            if new_mk < current_mk:
                current_perm = new_perm
                current_C = new_C
                current_mk = new_mk
                improved = True
                
                if current_mk < best_mk:
                    best_mk = current_mk
                    best_perm = current_perm.copy()
                    
    return list(best_perm)
# EVOLVE-BLOCK-END
