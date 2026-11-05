"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 140.0  # Safety margin before 150s wall-clock kill
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: CPM-based Total Float (slack) as primary driver,
    combined with resource contention metrics.
    
    Score = w1*Float + w2*RP + w3*MRP + w4*Dur
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback: return simple duration-based scores
        durations = np.asarray(instance["durations"], dtype=float)
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph and compute CPM values
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycle exists, fallback to index order
    if len(top_order) < n:
        top_order = list(range(n))

    # Compute Early Start (ES) and Early Finish (EF) - forward pass
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        for v in successors[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
    
    # Compute Late Finish (LF) and Late Start (LS) - backward pass
    project_end = np.max(ef) if n > 0 else 0.0
    lf = np.full(n, project_end)
    ls = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            if lf[u] > ls[v]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
    
    # Total Float (Slack) = LS - ES = LF - EF
    total_float = ls - es
    
    # Bottom level (longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate

    # 2. Compute Resource Metrics
    if demands is None or capacities is None:
        # Fallback: Just negative float (lower float = higher priority)
        float_max = np.max(total_float) if np.max(total_float) > 0 else 1.0
        return -total_float / float_max

    # Process demands into matrix
    if isinstance(demands, list) and len(demands) > 0:
        if isinstance(demands[0], dict):
            n_resources = len(capacities)
            demand_matrix = np.zeros((n, n_resources), dtype=float)
            for i, d in enumerate(demands):
                for r, val in d.items():
                    if r < n_resources:
                        demand_matrix[i, r] = val
        elif isinstance(demands[0], (list, tuple, np.ndarray)):
            demand_matrix = np.asarray(demands, dtype=float)
            if demand_matrix.shape[0] != n:
                try:
                    demand_matrix = demand_matrix.reshape(n, -1)
                except:
                    demand_matrix = np.zeros((n, 1))
        else:
            demand_matrix = np.zeros((n, 1), dtype=float)
    else:
        demand_matrix = np.zeros((n, 1), dtype=float)
        
    capacities_arr = np.asarray(capacities, dtype=float)
    if len(capacities_arr) == 0:
        capacities_arr = np.array([1.0])
        
    # Ensure dimensions match
    if demand_matrix.shape[1] != len(capacities_arr):
        new_dim = min(demand_matrix.shape[1], len(capacities_arr))
        demand_matrix = demand_matrix[:, :new_dim]
        capacities_arr = capacities_arr[:new_dim]
        
    safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
    
    # Resource Pressure: Sum of ratios
    resource_pressure = np.sum(demand_matrix / safe_caps, axis=1)
    # Max Resource Pressure: Max of ratios
    max_resource_pressure = np.max(demand_matrix / safe_caps, axis=1)

    # 3. Compute simulation-based resource contention
    # Simulate serial SGS with current priorities to identify which activities
    # cause the most delays due to resource conflicts
    
    # First, create an initial priority ordering based on float
    float_min = np.min(total_float)
    float_range = np.max(total_float) - float_min
    if float_range > 0:
        float_norm = 1.0 - (total_float - float_min) / float_range
    else:
        float_norm = np.ones(n)
    
    # Initial scores for simulation
    initial_scores = float_norm.copy()
    
    # Simulate serial SGS to measure actual delays caused by each activity
    # This helps identify activities that, when scheduled early, prevent cascading delays
    
    # Build eligibility and resource tracking
    remaining_preds = [len(pred_lists[i]) for i in range(n)]
    scheduled = [False] * n
    start_times = [0.0] * n
    resource_usage = np.zeros(len(capacities_arr), dtype=float)
    
    # Activity queue sorted by priority (descending)
    eligible = [i for i in range(n) if remaining_preds[i] == 0]
    eligible.sort(key=lambda x: -initial_scores[x])
    
    # Track which activities caused delays
    delay_causes = np.zeros(n, dtype=float)
    current_time = 0.0
    
    # Simple simulation: schedule activities in priority order, tracking resource conflicts
    while eligible:
        # Find the highest-priority eligible activity that can be scheduled
        scheduled_any = False
        for idx, act in enumerate(eligible):
            # Check if resources are available
            can_schedule = True
            for r in range(len(capacities_arr)):
                if demand_matrix[act, r] > resource_usage[r]:
                    can_schedule = False
                    break
            
            if can_schedule:
                # Schedule this activity
                scheduled[act] = True
                start_times[act] = current_time
                for r in range(len(capacities_arr)):
                    resource_usage[r] += demand_matrix[act, r]
                
                # Update successors
                for v in successors[act]:
                    remaining_preds[v] -= 1
                    if remaining_preds[v] == 0:
                        eligible.append(v)
                        eligible.sort(key=lambda x: -initial_scores[x])
                
                # Remove from eligible
                eligible.pop(idx)
                scheduled_any = True
                break
        
        if not scheduled_any:
            # Need to wait for resources to free up
            # Find the earliest finishing activity
            min_end = float('inf')
            for i in range(n):
                if scheduled[i]:
                    end = start_times[i] + durations[i]
                    if end < min_end:
                        min_end = end
            
            if min_end == float('inf'):
                break
            
            current_time = min_end
            # Free up resources for completed activities
            resource_usage = np.zeros(len(capacities_arr), dtype=float)
            for i in range(n):
                if scheduled[i] and start_times[i] + durations[i] <= current_time:
                    for r in range(len(capacities_arr)):
                        resource_usage[r] -= demand_matrix[i, r]
                    resource_usage = np.where(resource_usage < 0, 0, resource_usage)
    
    # Alternative approach: Use a simpler contention metric based on resource overlap
    # Count how many other activities compete for the same resources
    contention = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if activities i and j share resources
            shared = np.sum((demand_matrix[i] > 0) & (demand_matrix[j] > 0))
            if shared > 0:
                # Weight by the intensity of competition
                contention[i] += shared * np.min(demand_matrix[i], demand_matrix[j]).sum() / (np.max(demand_matrix[i]).sum() + 1e-9)
    
    # Normalize contention
    cont_max = np.max(contention) if np.max(contention) > 0 else 1.0
    cont_norm = contention / cont_max
    
    # Normalize other metrics
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    mrp_max = np.max(max_resource_pressure) if np.max(max_resource_pressure) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0

    rp_norm = resource_pressure / rp_max
    mrp_norm = max_resource_pressure / mrp_max
    dur_norm = durations / dur_max

    # Weights: Float is primary, contention captures dynamic resource conflicts
    w_float = 1.0
    w_rp = 0.25
    w_mrp = 0.1
    w_dur = 0.05
    w_cont = 0.3

    scores = w_float * float_norm + w_rp * rp_norm + w_mrp * mrp_norm + w_dur * dur_norm + w_cont * cont_norm

    return scores
# EVOLVE-BLOCK-END
