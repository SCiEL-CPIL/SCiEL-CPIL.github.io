"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """
    Compute makespan for a given job order.
    Optimized for speed in Python by using local variables and minimal overhead.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    completion = [0.0] * n_machines
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            if prev < completion[m]:
                prev = completion[m]
            prev += p[m]
            completion[m] = prev
    return completion[-1]


def _compute_edge_weights(processing_times):
    """
    Compute edge weights approximating incremental makespan contribution.
    For job i followed by job j, the weight is the additional makespan
    when j is placed immediately after i.
    """
    n_jobs, n_machines = processing_times.shape
    weights = np.zeros((n_jobs, n_jobs))
    
    for i in range(n_jobs):
        for j in range(n_jobs):
            if i == j:
                weights[i, j] = float('inf')
                continue
            # Simulate: place i then j, compute makespan
            order = [i, j]
            completion = [0.0] * n_machines
            for job in order:
                p = processing_times[job]
                prev = 0.0
                for m in range(n_machines):
                    if prev < completion[m]:
                        prev = completion[m]
                    prev += p[m]
                    completion[m] = prev
            weights[i, j] = completion[-1]
    
    return weights


def _lagrangian_greedy(processing_times, edge_weights, seed=None, penalty_scale=1.0):
    """
    Lagrangian Relaxation heuristic: use edge weights with penalty multipliers
    to guide a greedy construction. The penalty encourages balanced completion
    times across machines by penalizing jobs that cause large imbalances.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Initialize penalty multipliers (one per machine)
    lambdas = np.zeros(n_machines)
    
    # Greedy construction: at each step, select the job that minimizes
    # adjusted cost = edge_weight + sum_m lambda_m * (completion_m - mean_completion)
    seq = []
    remaining = set(range(n_jobs))
    
    # Track current completion times
    completion = [0.0] * n_machines
    
    for step in range(n_jobs):
        if len(remaining) == 1:
            job = list(remaining)[0]
            seq.append(job)
            remaining.remove(job)
            # Update completion
            p = processing_times[job]
            for m in range(n_machines):
                if completion[m] < 0:
                    completion[m] = 0.0
                completion[m] += p[m]
            continue
        
        best_job = None
        best_cost = float('inf')
        
        for job in remaining:
            # Compute cost: base edge weight (use average incoming edge weight as proxy)
            # + penalty term based on machine imbalance
            p = processing_times[job]
            
            # Simulate adding this job to current completion
            new_completion = completion[:]
            for m in range(n_machines):
                if new_completion[m] < 0:
                    new_completion[m] = 0.0
                new_completion[m] += p[m]
            
            # Penalty: sum of lambda_m * max(0, new_completion[m] - mean_completion)
            mean_comp = np.mean(new_completion)
            penalty = sum(lambdas[m] * max(0, new_completion[m] - mean_comp) for m in range(n_machines))
            
            # Base cost: use the makespan if this were the last job
            base_cost = max(new_completion)
            
            total_cost = base_cost + penalty_scale * penalty
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_job = job
        
        if best_job is None:
            break
        
        seq.append(best_job)
        remaining.remove(best_job)
        
        # Update completion
        p = processing_times[best_job]
        for m in range(n_machines):
            if completion[m] < 0:
                completion[m] = 0.0
            completion[m] += p[m]
        
        # Update lambdas: increase for machines that are "lagging" or "leading"
        # to encourage balance
        mean_comp = np.mean(completion)
        for m in range(n_machines):
            if completion[m] > mean_comp:
                lambdas[m] += 0.1
            else:
                lambdas[m] = max(0, lambdas[m] - 0.05)
    
    return seq


def _neh_initialization(processing_times, seed=None):
    """
    NEH heuristic with randomization for diversity.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Order jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    
    # Add randomization to break ties and create diversity
    range_totals = totals.max() - totals.min()
    if range_totals < 1e-9:
        range_totals = 1.0
        
    perturbation = rng.rand(n_jobs) * 0.1 * range_totals
    perturbed_totals = totals + perturbation
    
    job_order = list(np.argsort(-perturbed_totals))
    
    # Build sequence greedily using NEH rule
    seq = [job_order[0]]
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _local_search(order, processing_times, time_limit=50.0):
    """
    Local search using insertion moves with First Improvement strategy.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, processing_times)
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                return best_order, best_ms
            
            job = best_order[i]
            
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                if j < i:
                    new_order = best_order[:j] + [job] + best_order[j:i] + best_order[i+1:]
                else:
                    new_order = best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:]
                
                ms = _makespan(new_order, processing_times)
                
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
            
            if improved:
                pass

    return best_order, best_ms


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses NEH restarts, Lagrangian greedy, and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Precompute edge weights once (for Lagrangian heuristic)
    edge_weights = _compute_edge_weights(processing_times)
    
    best_order = None
    best_ms = float('inf')
    
    # Determine number of restarts
    n_restarts = 15 if n_jobs < 30 else 8
    
    for r in range(n_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        seed = r * 12345 + 99
        
        # Alternate between NEH and Lagrangian greedy for diversity
        if r % 2 == 0:
            initial = _neh_initialization(processing_times, seed=seed)
        else:
            # Use different penalty scales for diversity
            penalty_scale = 0.5 + 0.5 * (r % 3)
            initial = _lagrangian_greedy(processing_times, edge_weights, seed=seed, penalty_scale=penalty_scale)
        
        # Apply local search
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            improved, ms = _local_search(initial, processing_times, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    if best_order is None:
        best_order = _neh_initialization(processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
