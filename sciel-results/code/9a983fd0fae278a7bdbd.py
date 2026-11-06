"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    """
    Compute a priority score for each activity in a resource-constrained project scheduling problem.
    Higher priority means the activity should be scheduled earlier in the serial SGS.
    
    The priority is based on:
    1. Total float (slack) - activities with less slack are more critical
    2. Resource demand intensity - activities consuming scarce resources
    3. Precedence depth - activities that are bottlenecks in the network
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    
    # If predecessors not provided, assume no precedence constraints
    if predecessors is None:
        pred_list = [set() for _ in range(n)]
    else:
        # predecessors[i] is a list/set of activities that must precede i
        pred_list = [set(p) for p in predecessors]
    
    # Build successor list
    succ_list = [set() for _ in range(n)]
    for i in range(n):
        for j in pred_list[i]:
            if i not in succ_list[j]:
                succ_list[j].add(i)
    
    # Compute earliest start times (ES) via forward pass
    es = np.zeros(n, dtype=float)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(pred_list[i])
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        u = queue.popleft()
        for v in succ_list[u]:
            es[v] = max(es[v], es[u] + durations[u])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute latest start times (LS) via backward pass
    # First find the project duration (max ES + duration)
    makespan = np.max(es + durations) if n > 0 else 0.0
    
    ls = np.full(n, makespan, dtype=float)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(succ_list[i])
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    while queue:
        u = queue.popleft()
        for v in pred_list[u]:
            ls[v] = min(ls[v], ls[u] - durations[v])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Total float = LS - ES (lower is more critical)
    total_float = ls - es
    
    # Compute resource scarcity: for each resource, what fraction of activities use it
    # and how much demand they have relative to capacity
    resource_score = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # demands_arr shape: (n, num_resources) or similar
        # capacities_arr shape: (num_resources,)
        
        if demands_arr.ndim == 1:
            # Single resource case
            if capacities_arr.ndim == 0:
                cap = float(capacities_arr)
                # Scarcity: demand / capacity
                resource_score = demands_arr / max(cap, 1e-9)
            else:
                cap = float(capacities_arr[0]) if len(capacities_arr) > 0 else 1.0
                resource_score = demands_arr / max(cap, 1e-9)
        else:
            # Multiple resources
            num_resources = demands_arr.shape[1] if demands_arr.ndim > 1 else 1
            for r in range(num_resources):
                cap_r = float(capacities_arr[r]) if r < len(capacities_arr) else 1.0
                # Weight by scarcity: higher demand relative to capacity means more urgent
                resource_score += demands_arr[:, r] / max(cap_r, 1e-9)
    
    # Combine: prioritize activities with low total float (critical) and high resource demand
    # Normalize each component
    if n > 0:
        tf_min = np.min(total_float)
        tf_max = np.max(total_float)
        tf_range = tf_max - tf_min if tf_max > tf_min else 1.0
        # Lower float -> higher priority, so invert
        float_priority = 1.0 - (total_float - tf_min) / tf_range
        
        rs_min = np.min(resource_score)
        rs_max = np.max(resource_score)
        rs_range = rs_max - rs_min if rs_max > rs_min else 1.0
        resource_priority = (resource_score - rs_min) / rs_range
        
        # Duration priority: longer activities first (to get them done early)
        d_min = np.min(durations)
        d_max = np.max(durations)
        d_range = d_max - d_min if d_max > d_min else 1.0
        duration_priority = (durations - d_min) / d_range
        
        # Weighted combination
        # Critical path activities (low float) should be scheduled first
        # High resource demand activities should be scheduled first to avoid bottlenecks
        # Long activities should be scheduled first to reduce makespan
        priority_scores = 0.5 * float_priority + 0.3 * resource_priority + 0.2 * duration_priority
    else:
        priority_scores = np.zeros(0, dtype=float)
    
    return priority_scores
# EVOLVE-BLOCK-END
