import numpy as np
from scipy.optimize import differential_evolution

def _calc_makespan(perm, processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        C_curr = np.zeros(n_machines)
        for m in range(n_machines):
            start = max(C_prev[m], C_curr[m-1] if m > 0 else 0)
            C_curr[m] = start + processing_times[job_idx, m]
        C_prev = C_curr
    return C_prev[-1]

def _local_search_insertion(perm, processing_times, max_iterations=100):
    """
    Simple insertion local search to improve the permutation.
    """
    n_jobs = len(perm)
    current_makespan = _calc_makespan(perm, processing_times)
    best_perm = list(perm)
    best_makespan = current_makespan
    improved = True
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                # Move job at i to position j
                new_perm = list(best_perm)
                job = new_perm.pop(i)
                # If i < j, the pop shifts indices, so j-1 is the correct insertion point?
                # Actually, if we pop i, the element at j (if j>i) moves to j-1.
                # We want to insert at position j in the *original* list?
                # Standard insertion move: remove job at i, insert at j.
                # If j > i, the target index in the new list is j-1.
                # If j < i, the target index is j.
                insert_idx = j - 1 if j > i else j
                new_perm.insert(insert_idx, job)
                
                if new_perm == best_perm:
                    continue
                    
                new_makespan = _calc_makespan(new_perm, processing_times)
                if new_makespan < best_makespan:
                    best_perm = new_perm
                    best_makespan = new_makespan
                    improved = True
                    # Break inner loop to restart with new best? Or continue?
                    # Restarting is safer for local search stability.
                    break
            if improved:
                break
                
    return best_perm

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # 1. Use DE to find a good weight vector
    def objective(weights):
        perm = np.argsort(-weights, kind='stable')
        return _calc_makespan(perm, processing_times)
        
    bounds = [(0, 1)] * n_jobs
    
    best_weights = None
    try:
        # Small population for speed
        result = differential_evolution(
            objective, 
            bounds, 
            popsize=8, 
            maxiter=20, 
            tol=1e-3,
            seed=42,
            init='sobol'
        )
        best_weights = result.x
    except Exception:
        # Fallback
        best_weights = np.random.rand(n_jobs)
        
    # 2. Decode weights to permutation
    perm = np.argsort(-best_weights, kind='stable')
    
    # 3. Local Search to polish
    # Limit local search time/iterations to ensure < 120s
    # N is likely small, so 100 iterations is fine.
    final_perm = _local_search_insertion(perm, processing_times, max_iterations=50)
    
    return list(final_perm)
