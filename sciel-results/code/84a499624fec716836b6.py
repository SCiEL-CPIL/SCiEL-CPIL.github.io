<<<<<<< SEARCH
def _makespan(order, p):
    """Compute makespan for a given job order on a flow-shop."""
    n = len(order)
    if n == 0:
        return 0.0
    # completion[m][j] = completion time of job j on machine m
    # We process jobs in the given order
    comp = np.zeros((p.shape[1], n), dtype=np.float64)
    for j in range(n):
        job = order[j]
        for m in range(p.shape[1]):
            if m == 0:
                comp[m, j] = p[job, m]
            else:
                comp[m, j] = max(comp[m-1, j], comp[m, j-1]) + p[job, m]
    return comp[-1, -1]
=======
def _makespan(order, p):
    """Compute makespan for a given job order on a flow-shop using vectorization."""
    n = len(order)
    if n == 0:
        return 0.0
    # p is (n_jobs, n_machines). order is a list of job indices.
    # We need to compute the completion time for each job on each machine.
    # C[i][m] = C[i][m-1] + P[order[i]][m]  for m=0
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[order[i]][m] for m>0
    
    # Extract processing times for the specific order
    # p_ordered has shape (n, m)
    p_ordered = p[order]
    
    # Initialize completion times
    # We can do this iteratively over machines or jobs. 
    # Vectorizing over jobs is tricky due to dependencies.
    # However, we can unroll the machine loop if m is small, or use a simple loop over jobs which is O(n*m).
    # Given the previous failure was likely O(n*m) in Python with high overhead, 
    # a NumPy-based loop over jobs (n iterations) where each step is O(m) vectorized might be better,
    # OR we can use the fact that m is usually small.
    
    # Let's try a loop over jobs, but using NumPy for the row update.
    # C[i] is a vector of length m.
    # C[0] = cumsum(p_ordered[0])
    # C[i] = cumsum(p_ordered[i]) + max(C[i-1], C[i][:-1] - p_ordered[i][1:]) ... this is getting complex.
    
    # Simpler fast approach:
    # The makespan is the sum of processing times on the critical path.
    # For flow shop, C[i][m] = max(C[i-1][m], C[i][m-1]) + P[i][m]
    
    # We will loop over jobs (n), but keep the machine calculation vectorized.
    # This is O(n*m) operations but with low constant factor.
    if n == 1:
        return np.sum(p_ordered[0])
        
    # C_prev is the completion times for the previous job (length m)
    # C_curr is the completion times for the current job (length m)
    C_prev = np.zeros(p.shape[1], dtype=np.float64)
    
    for i in range(n):
        p_i = p_ordered[i]
        # C_curr[m] = max(C_prev[m], C_curr[m-1]) + p_i[m]
        # We can compute this iteratively over m, but m is usually small.
        # Or we can use a trick:
        # Let A[m] = C_prev[m]
        # Let B[m] = C_curr[m-1] (with B[0] = 0)
        # C_curr[m] = max(A[m], B[m]) + p_i[m]
        
        # Since m is small, a Python loop over m is often faster than complex NumPy indexing overhead
        # but let's try to be as vectorized as possible.
        # Actually, for small m, a simple loop is fine. The previous code had a loop over n AND m.
        # If we keep the loop over n, we save the outer loop overhead? No.
        
        # Let's stick to the most reliable fast method:
        # Loop over jobs, vectorize machine dependencies if possible.
        # For flow shop, there is no simple single-vector formula for all jobs.
        
        # Let's just optimize the existing logic.
        # The previous code:
        # for j in range(n):
        #   for m in range(m):
        # This is slow in Python.
        
        # New approach:
        # We can compute the entire matrix C using a loop over jobs, 
        # but update the row using NumPy operations.
        
        # C_curr[0] = C_prev[0] + p_i[0]  (Wait, C_prev[0] is completion of job i-1 on machine 0)
        # Actually, C[i][0] = C[i-1][0] + P[i][0]
        # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[i][m]
        
        C_curr = np.empty(p.shape[1], dtype=np.float64)
        C_curr[0] = C_prev[0] + p_i[0]
        if p.shape[1] > 1:
            # We need to compute C_curr[1:]
            # C_curr[m] depends on C_curr[m-1]
            # This sequential dependency prevents full vectorization over m.
            # However, m is typically small (e.g., 5-10).
            # So a loop over m is acceptable, but we can use NumPy for the max if we unroll?
            # No, let's just use a tight Python loop for m, it's fast enough if n is the main driver.
            # Wait, the previous code had nested loops. 
            # If n=100, m=10, that's 1000 ops. That's fast.
            # Why did it time out? 
            # Ah, `_two_opt` calls `_makespan` O(n^2) times. 
            # If n=100, n^2=10000 calls. 10000 * 1000 ops = 10^7 ops. Should be < 1s.
            # Maybe n is larger? Or the overhead of list copying `candidate = current[:]` was high?
            
            # Let's optimize the makespan calculation itself to be as fast as possible.
            # We can pre-allocate C matrix? No, it changes every call.
            
            # Let's use a slightly optimized loop:
            for m in range(1, p.shape[1]):
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + p_i[m]
        C_prev = C_curr
        
    return C_prev[-1]
>>>>>>> REPLACE

