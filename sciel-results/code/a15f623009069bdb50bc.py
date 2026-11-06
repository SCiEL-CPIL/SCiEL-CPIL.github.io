# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Uses a combination of critical path length (backward pass), resource pressure,
    and duration to produce a robust priority list.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin below 120s limit

    # Extract instance data
    N = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)

    # Handle predecessors
    # Assume instance["predecessors"] is a list of lists, where preds[i] is list of indices of predecessors of i
    # Or it might be a dict. Let's handle both common formats.
    if "predecessors" in instance:
        preds = instance["predecessors"]
        if isinstance(preds, dict):
            preds_list = [preds.get(i, []) for i in range(N)]
        else:
            preds_list = preds
    else:
        preds_list = [[] for _ in range(N)]

    # Build successors list
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds_list[i]:
            succs[p].append(i)

    # Compute backward pass (critical path from each node to end)
    # We need topological order. Since it's a DAG, we can use Kahn's algorithm or DFS.
    # For simplicity and speed with N=30, we can do a simple recursive/iterative topological sort.
    
    # Compute in-degrees
    in_degree = [0] * N
    for i in range(N):
        for p in preds_list[i]:
            in_degree[i] += 1

    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(N) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)

    # If topological sort didn't include all nodes, there's a cycle (shouldn't happen)
    # But let's be safe
    if len(topo_order) < N:
        # Fallback: just use duration
        return durations

    # Backward pass: compute longest path from each node to any sink
    # Process in reverse topological order
    backward = np.zeros(N, dtype=float)
    for i in reversed(topo_order):
        max_succ = 0
        for v in succs[i]:
            max_succ = max(max_succ, backward[v])
        backward[i] = durations[i] + max_succ

    # Compute resource pressure using MAX ratio (captures bottleneck resource)
    resource_pressure = np.zeros(N, dtype=float)
    for i in range(N):
        # demands[i] is the vector of resource demands for activity i
        # capacities is the vector of resource capacities
        # Avoid division by zero
        ratios = np.where(capacities > 0, demands[i] / np.where(capacities > 0, capacities, 1.0), 0)
        resource_pressure[i] = np.max(ratios)

    # Combine priorities
    # We want higher score = higher priority
    # Let's normalize each component to [0, 1] or just use raw values with weights
    # Critical path (backward) is a strong indicator
    # Duration is also important
    # Resource pressure: activities that use more resources relative to capacity might be harder to schedule, so maybe they should have higher priority? Or lower? 
    # In scheduling, often you want to schedule resource-intensive tasks early to avoid blocking. Let's assume higher resource pressure -> higher priority.
    
    # Let's use a weighted sum
    # Normalize backward and duration to [0, 1] for combination
    b_min, b_max = np.min(backward), np.max(backward)
    b_range = b_max - b_min if b_max > b_min else 1.0
    b_norm = (backward - b_min) / b_range

    d_min, d_max = np.min(durations), np.max(durations)
    d_range = d_max - d_min if d_max > d_min else 1.0
    d_norm = (durations - d_min) / d_range

    rp_min, rp_max = np.min(resource_pressure), np.max(resource_pressure)
    rp_range = rp_max - rp_min if rp_max > rp_min else 1.0
    rp_norm = (resource_pressure - rp_min) / rp_range

    # Weights: critical path dominates, resource pressure second, duration as tiebreaker
    # Use a smaller weight for duration so it acts more as a tiebreaker
    weights = (0.7, 0.2, 0.1)
    priority_scores = weights[0] * b_norm + weights[1] * rp_norm + weights[2] * d_norm
    
    # Add a small negative bonus for shorter duration to break ties in favor of shorter tasks
    # This helps reduce idle time gaps when critical path lengths are similar
    duration_tiebreak = -0.05 * d_norm
    priority_scores += duration_tiebreak

    return priority_scores
# EVOLVE-BLOCK-END
