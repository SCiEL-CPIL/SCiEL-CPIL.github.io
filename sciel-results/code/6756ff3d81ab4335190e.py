"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 140.0

def _serial_sgs(durations, demands, capacities, predecessors, priority_vec):
    """
    Simulate the Serial Schedule Generation Scheme (SGS).
    Returns the makespan.
    """
    n = len(durations)
    
    # Calculate remaining predecessors count for each activity
    remaining_preds = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j] is not None:
            remaining_preds[j] = len(predecessors[j])
            
    # Precompute successor lists for quick updates
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j] is not None:
            for i in predecessors[j]:
                successors[i].append(j)

    # State
    start_times = np.zeros(n, dtype=float)
    finish_times = np.zeros(n, dtype=float)
    scheduled = np.zeros(n, dtype=bool)
    
    # Resource usage over time
    # To avoid O(T*n) loops, we track resource usage at event points.
    # However, for simplicity and robustness in a 120s limit with n=30, 
    # a simple time-stepping approach might be too slow if T is large.
    # But n=30 is small. Let's use a list of active jobs and their finish times.
    
    active_jobs = [] # List of (finish_time, resource_demands_dict)
    current_time = 0.0
    
    # Priority queue for eligible jobs
    # We need to pick the highest priority eligible job.
    # Eligible: all preds scheduled.
    
    # For efficiency, we can maintain a list of eligible jobs and pick the max priority.
    # Since n=30, O(n^2) is fine.
    
    for _ in range(n):
        # Find eligible jobs
        eligible = []
        for j in range(n):
            if not scheduled[j] and remaining_preds[j] == 0:
                eligible.append(j)
        
        if not eligible:
            # Should not happen if DAG is valid
            break
            
        # Select job with highest priority
        # priority_vec is higher = higher priority
        best_job = max(eligible, key=lambda j: priority_vec[j])
        
        # Determine earliest start time for best_job
        earliest_start = current_time
        
        # Check precedence constraints (already satisfied by eligibility)
        # Check resource constraints
        # We need to find the earliest time t >= earliest_start such that
        # resources are available for best_job.
        
        # Current resource usage at earliest_start
        # We need to simulate time forward until resources are free.
        
        # Optimization: We can look at the finish times of active jobs.
        # The resource usage only changes when a job finishes.
        # So candidate start times are: earliest_start, and finish times of active jobs.
        
        candidate_times = [earliest_start]
        for job_idx, finish_t, demands in active_jobs:
            if finish_t > earliest_start:
                candidate_times.append(finish_t)
        
        candidate_times = sorted(set(candidate_times))
        
        start_time = earliest_start
        feasible = False
        
        # Check if resources are available at earliest_start
        # We need to sum demands of active jobs that are still running at earliest_start
        # Plus the demand of best_job
        
        # Actually, it's easier to just check each candidate time.
        # At a candidate time t, which jobs are active?
        # Jobs with finish_time > t are active.
        
        for t in candidate_times:
            # Check if resources are available at t
            res_usage = np.zeros(len(capacities))
            for job_idx, finish_t, demands in active_jobs:
                if finish_t > t:
                    res_usage += demands
            
            # Check if best_job fits
            if np.all(res_usage + demands[best_job] <= capacities):
                start_time = t
                feasible = True
                break
        
        if not feasible:
            # This shouldn't happen if capacities are sufficient for individual jobs
            # but if a job alone exceeds capacity, it's infeasible.
            # Assuming instances are feasible.
            start_time = earliest_start
            
        # Schedule the job
        scheduled[best_job] = True
        finish_time = start_time + durations[best_job]
        start_times[best_job] = start_time
        finish_times[best_job] = finish_time
        
        # Update active jobs
        active_jobs.append((finish_time, demands[best_job]))
        
        # Update current_time to at least start_time
        if start_time > current_time:
            current_time = start_time
            
        # Update remaining preds for successors
        for succ in successors[best_job]:
            remaining_preds[succ] -= 1
            
    return np.max(finish_times)

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]
    capacities = instance["capacities"]
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # Candidate priority rules
    rules = {
        "LPT": durations,  # Longest Processing Time
        "SPT": -durations, # Shortest Processing Time (negate for max)
        "MIN_SLACK": np.zeros(n), # Placeholder, calculated below
    }
    
    # Calculate Slack (Total Slack)
    # ES (Earliest Start)
    es = np.zeros(n)
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j] is not None:
            in_degree[j] = len(predecessors[j])
            for i in predecessors[j]:
                successors[i].append(j)
                
    queue = [i for i in range(n) if in_degree[i] == 0]
    topological_order = []
    while queue:
        u = queue.pop(0)
        topological_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    for u in topological_order:
        for v in successors[u]:
            es[v] = max(es[v], es[u] + durations[u])
            
    # LF (Latest Finish)
    lf = np.zeros(n)
    project_duration = np.max(es + durations)
    for u in reversed(topological_order):
        if not successors[u]:
            lf[u] = project_duration
        else:
            lf[u] = min([lf[v] - durations[v] for v in successors[u]])
            
    slack = lf - (es + durations)
    rules["MIN_SLACK"] = -slack # Higher priority for lower slack
    
    # Add a random rule for diversity? Or maybe Critical Path based?
    # Let's stick to these 3 strong heuristics.
    
    best_priority = None
    best_makespan = float('inf')
    
    for name, p_vec in rules.items():
        # Check time
        if time.time() - _START_TIME > _DEADLINE_SECONDS - 5:
            break
            
        makespan = _serial_sgs(durations, demands, capacities, predecessors, p_vec)
        if makespan < best_makespan:
            best_makespan = makespan
            best_priority = p_vec
            
    # If no rule was evaluated due to time, fallback to LPT
    if best_priority is None:
        best_priority = rules["LPT"]
        
    return best_priority
# EVOLVE-BLOCK-END
