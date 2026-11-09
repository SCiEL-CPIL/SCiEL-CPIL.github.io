"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

_DEADLINE_SECONDS = 145.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def _run_serial_sgs_fast(instance, priority_vec):
    """
    Optimized Serial SGS.
    Uses a pre-computed list of resource users for each resource to speed up conflict checking.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return 0.0
        
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # Pre-compute resource users
    # resource_users[r] = list of job indices that use resource r
    num_resources = 0
    if demands is not None:
        d_mat = np.asarray(demands)
        if len(d_mat.shape) == 2:
            num_resources = d_mat.shape[1]
    
    resource_users = [[] for _ in range(num_resources)]
    if demands is not None:
        d_mat = np.asarray(demands, dtype=float)
        for i in range(n):
            for r in range(num_resources):
                if d_mat[i][r] > 0:
                    resource_users[r].append(i)
                    
    # State
    scheduled = [False] * n
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    
    # Pre-compute successors for quick eligibility check? 
    # Or just check preds. n=30 is small.
    
    # Active jobs per resource for quick load check?
    # Since n is small, checking all scheduled jobs that use the resource is O(n) per resource.
    # Total O(n * R) per job. 30 * 4 = 120 ops per job. 30 jobs = 3600 ops. Very fast.
    
    scheduled_count = 0
    # We need to pick the highest priority eligible job.
    # Eligible: all preds scheduled.
    
    # To speed up selection, we could maintain a heap, but eligibility changes dynamically.
    # With n=30, scanning all n jobs to find eligible and max priority is O(n^2) total.
    
    while scheduled_count < n:
        best_job = -1
        best_prio = -np.inf
        
        # Find eligible job with max priority
        for i in range(n):
            if not scheduled[i]:
                # Check preds
                eligible = True
                for p in predecessors[i]:
                    if not scheduled[p]:
                        eligible = False
                        break
                if eligible:
                    if priority_vec[i] > best_prio:
                        best_prio = priority_vec[i]
                        best_job = i
                        
        if best_job == -1:
            # Should not happen in valid DAG
            break
            
        # Determine earliest start time
        # 1. Predecessor constraint
        es_pred = 0.0
        for p in predecessors[best_job]:
            if finish_times[p] > es_pred:
                es_pred = finish_times[p]
                
        # 2. Resource constraint
        # Candidate times: es_pred and finish times of jobs using same resources
        candidate_times = {es_pred}
        
        if demands is not None and capacities is not None:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
            
            for r in range(num_resources):
                if d_mat[best_job][r] > 0:
                    # Add finish times of all scheduled jobs using resource r
                    for j in resource_users[r]:
                        if scheduled[j]:
                            candidate_times.add(finish_times[j])
                            
        # Sort candidates
        earliest_start = float('inf')
        for t in sorted(candidate_times):
            if t < es_pred:
                continue
            
            # Check resource availability at t
            feasible = True
            if demands is not None and capacities is not None:
                d_mat = np.asarray(demands, dtype=float)
                c_vec = np.asarray(capacities, dtype=float)
                
                for r in range(num_resources):
                    if d_mat[best_job][r] == 0:
                        continue
                    load = 0.0
                    for j in resource_users[r]:
                        if scheduled[j]:
                            if start_times[j] <= t < finish_times[j]:
                                load += d_mat[j][r]
                    if load + d_mat[best_job][r] > c_vec[r]:
                        feasible = False
                        break
                        
            if feasible:
                earliest_start = t
                break
                
        if earliest_start == float('inf'):
            earliest_start = es_pred
            
        start_times[best_job] = earliest_start
        finish_times[best_job] = earliest_start + durations[best_job]
        scheduled[best_job] = True
        scheduled_count += 1
        
    makespan = np.max(finish_times) if n > 0 else 0.0
    return makespan

def _generate_initial_priorities(instance):
    """
    Generate a set of diverse initial priority vectors.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    n = len(durations)
    
    if n == 0:
        return []
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # 1. LPT
    p1 = durations.copy()
    
    # 2. Random
    p2 = np.random.rand(n)
    
    # 3. Critical Path (Slack)
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        ef[u] = es[u] + durations[u]
        
    makespan_no_res = np.max(ef)
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    slack = ls - es
    p3 = -slack
    
    # 4. Resource Intensity
    p4 = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        d_mat = np.asarray(demands, dtype=float)
        c_vec = np.asarray(capacities, dtype=float)
        p4 = np.sum(d_mat / c_vec, axis=1)
        
    # 5. Hybrid
    p5 = p1 + p3 * 0.5 + p4 * 0.5
    
    return [p1, p2, p3, p4, p5]

def priority(instance):
    """
    Uses local search (hill climbing) to find a good priority vector.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([])
        
    # Set random seed for reproducibility within the run if needed, but usually 
    # we want diversity. However, the evaluator might call this once per instance.
    # Using a fixed seed ensures deterministic behavior for the same instance.
    # Let's use a seed derived from the instance to be deterministic but instance-specific.
    # Or just use a fixed seed for simplicity in this context.
    np.random.seed(42)
    
    initial_vectors = _generate_initial_priorities(instance)
    
    best_vec = None
    best_score = float('inf')
    
    # Evaluate initial vectors
    for vec in initial_vectors:
        score = _run_serial_sgs_fast(instance, vec)
        if score < best_score:
            best_score = score
            best_vec = vec.copy()
            
    # Local Search
    # We will perform random perturbation and hill climbing.
    # Time budget check
    max_iterations = 10000
    step_size = 1.0
    
    for i in range(max_iterations):
        if _remaining_budget() < 5.0:
            break
            
        # Perturb current best vector
        # Add Gaussian noise
        noise = np.random.randn(n) * step_size
        new_vec = best_vec + noise
        
        # Optional: Normalize or clip? 
        # Priorities are relative, so scaling doesn't matter.
        # Shifting doesn't matter.
        
        score = _run_serial_sgs_fast(instance, new_vec)
        
        if score < best_score:
            best_score = score
            best_vec = new_vec
            # Maybe reduce step size slightly on improvement?
            # step_size *= 0.95
        else:
            # If we haven't improved in a while, maybe restart from a different initial?
            # Or just continue.
            pass
            
    return best_vec

# EVOLVE-BLOCK-END
