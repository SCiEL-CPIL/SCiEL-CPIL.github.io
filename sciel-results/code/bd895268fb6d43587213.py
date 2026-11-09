import numpy as np
from scipy.optimize import differential_evolution
import time

def _calc_makespan_from_weights(weights, processing_times):
    """
    Compute makespan from a weight vector by deriving the permutation.
    Jobs are ordered by descending weight (stable sort).
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    
    # Derive permutation from weights: higher weight = earlier position
    perm = np.argsort(-weights, kind='stable')
    
    C_prev = np.zeros(n_machines)
    for job_idx in perm:
        C_curr = np.zeros(n_machines)
        for m in range(n_machines):
            start = max(C_prev[m], C_curr[m-1] if m > 0 else 0)
            C_curr[m] = start + processing_times[job_idx, m]
        C_prev = C_curr
    return C_prev[-1]

def _heuristic_weights(processing_times):
    """
    Generate initial weight vector using heuristics.
    Combines total processing time and machine-specific features.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Base weight: sum of processing times (higher = more critical)
    sums = np.sum(processing_times, axis=1)
    
    # Add noise to break ties and explore
    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 0.1, size=n_jobs)
    
    weights = sums + noise
    
    # Normalize to [0, 1]
    if np.max(weights) > np.min(weights):
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    else:
        weights = np.linspace(0, 1, n_jobs)
    
    return weights

def _local_search_weights(weights, processing_times, time_limit=5.0):
    """
    Local search in weight space: perturb weights and accept if makespan improves.
    This is a continuous analogue of insertion moves.
    """
    n_jobs = len(weights)
    best_weights = weights.copy()
    best_makespan = _calc_makespan_from_weights(best_weights, processing_times)
    start_time = time.time()
    
    improved = True
    step_size = 0.05
    
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Try random perturbations
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Perturb a random subset of weights
            n_perturb = max(1, n_jobs // 5)
            indices = np.random.choice(n_jobs, size=min(n_perturb, n_jobs), replace=False)
            
            new_weights = best_weights.copy()
            for idx in indices:
                new_weights[idx] += np.random.uniform(-step_size, step_size)
                new_weights[idx] = np.clip(new_weights[idx], 0, 1)
            
            new_makespan = _calc_makespan_from_weights(new_weights, processing_times)
            
            if new_makespan < best_makespan:
                best_weights = new_weights
                best_makespan = new_makespan
                improved = True
                break
    
    return best_weights, best_makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # 1. Generate initial weight vector using heuristics
    initial_weights = _heuristic_weights(processing_times)
    makespan_initial = _calc_makespan_from_weights(initial_weights, processing_times)
    
    best_weights = initial_weights.copy()
    best_makespan = makespan_initial
    
    # 2. Use differential evolution to optimize weight vector
    # DE works well in continuous spaces
    if n_jobs <= 50:
        def objective(weights):
            return _calc_makespan_from_weights(weights, processing_times)
        
        bounds = [(0, 1)] * n_jobs
        
        try:
            # Use larger population and more iterations for better convergence
            popsize = max(10, min(20, 40 // max(1, n_jobs // 10)))
            maxiter = max(30, min(60, 300 // max(1, n_jobs // 5)))
            
            result = differential_evolution(
                objective, 
                bounds, 
                popsize=popsize, 
                maxiter=maxiter, 
                tol=1e-5,
                seed=42,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if result.success or result.fun < best_makespan:
                best_weights = result.x
                best_makespan = result.fun
        except Exception:
            pass  # Fallback to heuristic
    
    # 3. Local search in weight space to polish
    # Use remaining time budget
    elapsed = time.time() - (time.time() - 0)  # Placeholder, we track internally
    final_weights, _ = _local_search_weights(best_weights, processing_times, time_limit=8.0)
    
    # Convert final weights to permutation
    final_perm = np.argsort(-final_weights, kind='stable')
    
    return list(final_perm)
