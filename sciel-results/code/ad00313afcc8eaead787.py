# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Strategy: Combine multiple heuristics:
    1. Longest path to end (critical path) - activities on critical path first
    2. Resource demand intensity - activities with high resource needs first
    3. Duration - longer activities first
    4. Successor count - activities with many successors first
    5. Predecessor count - activities with few predecessors (can start early)
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    preds = instance.get("predecessors", None)
    
    # Build successor lists
    succs = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                for p in preds[i]:
                    if p >= 0:
                        succs[p].append(i)
    
    # Compute longest path from each node to end (reverse topological order)
    # First, compute in-degrees for topological sort
    in_degree = [0] * n
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                for p in preds[i]:
                    if p >= 0:
                        in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes were processed, add remaining (cycle handling)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Longest path from node to end
    longest_to_end = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        if succs[u]:
            max_succ = max(longest_to_end[v] for v in succs[u])
            longest_to_end[u] = durations[u] + max_succ
        else:
            longest_to_end[u] = durations[u]
    
    # Longest path from start to node
    longest_from_start = np.zeros(n, dtype=float)
    for u in topo_order:
        if preds is not None and preds[u] is not None and len(preds[u]) > 0:
            valid_preds = [p for p in preds[u] if p >= 0]
            if valid_preds:
                max_pred = max(longest_from_start[p] for p in valid_preds)
                longest_from_start[u] = max_pred + durations[u]
            else:
                longest_from_start[u] = durations[u]
        else:
            longest_from_start[u] = durations[u]
    
    # Resource demands
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None:
        demands = np.asarray(demands, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if capacities is not None:
            capacities = np.asarray(capacities, dtype=float)
            # Normalize demands by capacities
            demand_ratio = np.sum(demands / capacities, axis=1)
        else:
            demand_ratio = np.sum(demands, axis=1)
    else:
        demand_ratio = np.ones(n, dtype=float)
    
    # Successor count
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Predecessor count
    pred_count = np.zeros(n, dtype=float)
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                pred_count[i] = len([p for p in preds[i] if p >= 0])
    
    # Combine heuristics with weights
    # Normalize each component to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > 1e-10:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    norm_longest_end = normalize(longest_to_end)
    norm_longest_start = normalize(longest_from_start)
    norm_duration = normalize(durations)
    norm_demand = normalize(demand_ratio)
    norm_succ = normalize(succ_count)
    norm_pred = normalize(pred_count)
    
    # Weighted combination
    # Critical path (longest to end) is most important
    # Then duration, demand, successors
    weights = {
        'longest_end': 0.40,
        'duration': 0.20,
        'demand': 0.15,
        'succ': 0.10,
        'longest_start': 0.10,
        'pred': 0.05
    }
    
    score = (weights['longest_end'] * norm_longest_end +
             weights['duration'] * norm_duration +
             weights['demand'] * norm_demand +
             weights['succ'] * norm_succ +
             weights['longest_start'] * norm_longest_start +
             weights['pred'] * (1 - norm_pred))  # Fewer predecessors = higher priority
    
    # Check time
    if time.time() - start_time > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
