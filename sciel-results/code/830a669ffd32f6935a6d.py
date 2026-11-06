"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation using vectorized numpy operations.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        perm: list or array of job indices in order
    
    Returns:
        makespan (float)
    """
    if len(perm) == 0:
        return 0.0
    
    # Get the processing times for the jobs in the order of perm
    # shape: (n_jobs_in_perm, n_machines)
    p = processing_times[perm]
    
    n_machines = p.shape[1]
    
    if n_machines == 1:
        return np.sum(p)
    
    # We compute the completion time matrix C
    # C[i, j] = completion time of job i on machine j
    # C[i, 0] = C[i-1, 0] + P[i, 0]  => Cumulative sum along axis 0
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
    
    # Initialize C
    C = np.empty_like(p, dtype=np.float64)
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(p[:, 0])
    
    # For subsequent machines, we have a dependency on C[i, j-1] and C[i-1, j]
    # This dependency chain (i, j-1) -> (i, j) prevents full vectorization across machines in a single step
    # without iterative loops over machines. However, the loop is over M (usually small, e.g., 5-10),
    # while the inner operation is vectorized over N (jobs). This is much faster than looping over N*M in Python.
    
    for j in range(1, n_machines):
        # C[:, j] depends on C[:, j-1] (already computed) and C[:, j] (from previous row, i-1)
        # We can compute this iteratively over rows, but that's slow.
        # Is there a vectorized way?
        # C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
        # This is a 2D recurrence. It is not trivially vectorizable with standard numpy primitives
        # because C[i, j] depends on C[i, j] (same element, previous iteration) and C[i-1, j].
        # Wait, C[i-1, j] is known. C[i, j-1] is known.
        # So C[i, j] is determined by known values.
        # But C[i, j] is needed for C[i+1, j].
        # So we must iterate over i for each j.
        
        # However, we can optimize the inner loop in Python by using numpy arrays?
        # No, the dependency is sequential along i.
        
        # Alternative: Use a different formulation?
        # Actually, for small M, the overhead of the Python loop over M is small.
        # The inner loop over N is the bottleneck.
        # Can we vectorize the inner loop over N?
        # C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
        # Let A[i] = C[i-1, j] (known from previous row)
        # Let B[i] = C[i, j-1] (known from previous column)
        # Then C[i, j] = max(A[i], B[i]) + P[i, j]
        # But C[i, j] is also A[i+1] for the next iteration? No, A is fixed for the whole column j.
        # Wait. C[i-1, j] is the value computed in the previous row for the same column j.
        # So we are computing column j row by row.
        # c_prev = C[0, j-1] is not right.
        
        # Let's trace:
        # j=1
        # i=0: C[0,1] = max(C[-1,1] (0), C[0,0]) + P[0,1]
        # i=1: C[1,1] = max(C[0,1], C[1,0]) + P[1,1]
        # ...
        
        # This is a sequential dependency along i. We cannot vectorize over i directly with standard numpy
        # without a scan operation. Numpy doesn't have a 2D scan that respects this specific max-plus dependency
        # easily in one shot.
        
        # However, we can use a Python loop over i, but that's O(N*M) Python ops.
        # The original code did O(N*M) Python ops.
        # Is this faster?
        # Original:
        # for j in perm: (N)
        #   for m in range(M): (M)
        #     python ops
        # Total: N*M python ops.
        
        # My vectorized attempt:
        # for j in range(1, M): (M)
        #   for i in range(N): (N)
        #     python ops
        # Total: N*M python ops.
        
        # Same complexity! Why would it be faster?
        # 1. Memory access patterns: The original code accesses `processing_times[j]` (row) and `completion[m]` (small array).
        #    This new code accesses `p[i, j]`, `C[i-1, j]`, `C[i, j-1]`.
        #    `p` is contiguous if perm is sorted, but perm is arbitrary. So `p` is a gather.
        #    `C` is contiguous.
        
        # Let's try a different vectorization trick.
        # There is a known trick for flow shop makespan using cumulative sums if we transform the problem?
        # No, the max dependency makes it non-additive.
        
        # Let's stick to the loop but optimize the inner loop in Python.
        # Actually, we can use `np.maximum` if we can express the recurrence differently?
        # No.
        
        # Let's try to implement the inner loop in a way that minimizes overhead.
        # We will use a Python loop for i, but access numpy arrays.
        
        # Optimization: Pre-extract column j-1 and column j?
        prev_col = C[:, j-1]
        curr_col = C[:, j]
        prev_job_row = C[i-1, j]
        
        # Let's just use the standard loop but ensure we are using local variables for speed.
        prev_c = C[0, j-1] # C[0, j-1] is known. For i=0, C[-1, j] is 0.
        # Wait, for i=0:
        # C[0, j] = max(0, C[0, j-1]) + P[0, j]
        
        # Let's re-do the loop properly.
        # C[:, 0] is done.
        # For j in 1..M-1:
        #   prev_val = 0.0 # This represents C[i-1, j] for i=0
        #   for i in 0..N-1:
        #      # C[i, j] = max(prev_val, C[i, j-1]) + P[i, j]
        #      if prev_val > C[i, j-1]:
        #          val = prev_val + P[i, j]
        #      else:
        #          val = C[i, j-1] + P[i, j]
        #      C[i, j] = val
        #      prev_val = val
        
        # This is still O(N*M) python loops.
        # Is there a way to speed this up?
        # Maybe using `numba`? Not available.
        # Maybe using `scipy`? No.
        
        # Let's reconsider the original code.
        # Original code:
        # completion = np.zeros(M)
        # for j in perm:
        #   row = P[j]
        #   prev_c = 0
        #   for m in range(M):
        #      if m==0: new_c = completion[0] + row[0]
        #      else: new_c = max(completion[m], prev_c) + row[m]
        #      completion[m] = new_c
        #      prev_c = new_c
        
        # This accesses `completion` (size M) and `row` (size M).
        # My new code accesses `C` (size N*M) and `p` (size N*M).
        # The original code has better cache locality for the small `completion` array.
        # However, the original code has a Python loop over N (outer) and M (inner).
        # My code has a Python loop over M (outer) and N (inner).
        # If M is small (e.g., 5) and N is large (e.g., 100), the outer loop is small.
        # The inner loop is large.
        
        # Let's try to optimize the inner loop in my code by using numpy operations on slices?
        # No, the dependency is sequential.
        
        # Actually, there is a vectorized algorithm for this using "prefix max" if we rotate the perspective?
        # No, the max is between "previous job on same machine" and "same job on previous machine".
        
        # Let's stick to the loop but try to make it as fast as possible in Python.
        # We can use a list comprehension? No, sequential dependency.
        
        # Let's try a hybrid: Use the original `_calc_makespan` but with minor optimizations.
        # The original code is actually quite standard.
        
        # Wait, I can use `numba` if it were available, but it's not.
        
        # Let's try a different strategy: **Bitonic Sorting / Pairwise Swaps**?
        # No.
        
        # Let's go back to the SA idea. Even if the makespan calc is O(N*M) Python ops,
        # if we can do 1000 evals/sec, that's fine.
        # Original code: 100 jobs, 5 machines. 500 ops per eval.
        # 1000 evals/sec = 500,000 ops/sec. Python can do ~10-50M simple ops/sec.
        # So 1000 evals/sec is very conservative. We can likely do 5000-10000 evals/sec.
        
        # So, let's implement SA with the **original** makespan calc (or a slightly optimized version)
        # because it's simpler and robust.
        
        pass # Placeholder, we will implement the loop below.
        
        # Implementing the loop for column j
        # We need to be careful with indexing.
        
        # Let's just use the original logic but inside this function?
        # No, let's implement the column-wise loop here.
        
        # Optimization: Store references to numpy arrays in local variables
        p_col_j = p[:, j]
        c_col_j_minus_1 = C[:, j-1]
        c_col_j = C[:, j]
        
        # We need to iterate i from 0 to N-1
        # c_prev_row = 0.0
        # for i in range(N):
        #    # c_col_j[i] = max(c_prev_row, c_col_j_minus_1[i]) + p_col_j[i]
        #    # c_prev_row = c_col_j[i]
            
        # Can we vectorize this?
        # No.
        
        # Let's just do the loop.
        c_prev_row = 0.0
        # Using a local variable for the loop range
        for i in range(p.shape[0]):
            # Inline the max
            if c_prev_row > c_col_j_minus_1[i]:
                val = c_prev_row + p_col_j[i]
            else:
                val = c_col_j_minus_1[i] + p_col_j[i]
            c_col_j[i] = val
            c_prev_row = val

    return C[-1, -1]


def _v_shape_heuristic(processing_times):
    """
    Generate an initial permutation using the V-shaped heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Compute total processing time for each job
    totals = processing_times.sum(axis=1)
    # Sort jobs by total processing time in descending order
    sorted_jobs = np.argsort(-totals)
    
    # V-shape: Place jobs in a V-shape pattern
    # First half in ascending order, second half in descending order
    half = n_jobs // 2
    left = sorted_jobs[:half]
    right = sorted_jobs[half:]
    
    # Reverse the right part
    right = right[::-1]
    
    return list(left) + list(right)


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: V-shaped Heuristic
    current_perm = _v_shape_heuristic(processing_times)
    current_ms = _calc_makespan_vectorized(processing_times, current_perm)
    
    best_perm = current_perm
    best_ms = current_ms
    
    # 2. Tabu Search
    # Parameters
    tabu_tenure = 10
    max_iter = 100000
    
    # Tabu list: store recent moves to prevent cycling
    # A move is defined by (job_index, from_position, to_position)
    tabu_list = []
    tabu_counter = {}
    
    # Neighborhood: Move a block of 1 or 2 jobs to a random position
    # Or-opt style
    
    for k in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Generate neighbor
        # Choose block size 1 or 2
        block_size = random.choice([1, 2])
        if block_size == 1:
            i = random.randint(0, n_jobs - 1)
            job = current_perm[i]
            # Remove
            temp_perm = current_perm[:i] + current_perm[i+1:]
            # Insert at random pos
            if len(temp_perm) == 0:
                new_perm = [job]
            else:
                k_ins = random.randint(0, len(temp_perm) - 1)
                # Avoid inserting back to same place
                if k_ins == i or (i < len(temp_perm) and k_ins == i): # Rough check
                    # Just allow it, probability is low
                    pass
                new_perm = temp_perm[:k_ins] + [job] + temp_perm[k_ins:]
        else:
            if n_jobs < 2:
                continue
            i = random.randint(0, n_jobs - 2)
            block = current_perm[i:i+2]
            temp_perm = current_perm[:i] + current_perm[i+2:]
            if len(temp_perm) == 0:
                new_perm = block
            else:
                k_ins = random.randint(0, len(temp_perm) - 1)
                new_perm = temp_perm[:k_ins] + block + temp_perm[k_ins:]
                
        new_ms = _calc_makespan_vectorized(processing_times, new_perm)
        
        # Check if move is tabu
        move_key = (i, k_ins, block_size)
        is_tabu = move_key in tabu_counter and tabu_counter[move_key] > k - tabu_tenure
        
        # Accept if not tabu or if it improves best (aspiration criterion)
        if not is_tabu or new_ms < best_ms:
            current_perm = new_perm
            current_ms = new_ms
            
            # Update tabu list
            tabu_counter[move_key] = k
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm
        else:
            # If tabu and not improving, skip
            pass
        
        # Periodically clear tabu list to prevent it from growing too large
        if k % 1000 == 0:
            # Remove old entries
            for key in list(tabu_counter.keys()):
                if tabu_counter[key] < k - tabu_tenure:
                    del tabu_counter[key]
    
    return best_perm
# EVOLVE-BLOCK-END
