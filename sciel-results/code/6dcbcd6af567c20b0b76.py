import numpy as np
import time


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Build predecessor lists
    predecessors = instance.get("predecessors", [])
    # predecessors[i] is a list of activity indices that must complete before i starts
    
    # Compute backward critical path: longest path from each activity to the end
    # Use topological order (reverse)
    # First, compute topological order
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Backward pass: longest path from each node to end
    # cp[i] = durations[i] + max(cp[j] for j in successors[i]), or durations[i] if no successors
    cp = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for node in reversed(topo_order):
        max_succ = 0.0
        for s in successors[node]:
            if cp[s] > max_succ:
                max_succ = cp[s]
        cp[node] = durations[node] + max_succ
    
    # Resource demand factor: sum of (demand/r) for each resource
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        # demands shape: (n, num_resources), capacities shape: (num_resources,)
        if demands.ndim == 2:
            # For each activity, compute sum of demand_j / capacity_j
            # Avoid division by zero
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            resource_score = np.sum(demands / safe_cap, axis=1)
        else:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            resource_score = demands / safe_cap
    
    # Combine: primary is critical path length, secondary is resource demand
    # Normalize both to [0, 1] range for combination
    cp_min, cp_max = cp.min(), cp.max()
    if cp_max > cp_min:
        cp_norm = (cp - cp_min) / (cp_max - cp_min)
    else:
        cp_norm = np.ones(n)
    
    rs_min, rs_max = resource_score.min(), resource_score.max()
    if rs_max > rs_min:
        rs_norm = (resource_score - rs_min) / (rs_max - rs_min)
    else:
        rs_norm = np.zeros(n)
    
    # Weighted combination: critical path is dominant
    priority_scores = 0.9 * cp_norm + 0.1 * rs_norm
    
    return priority_scores
