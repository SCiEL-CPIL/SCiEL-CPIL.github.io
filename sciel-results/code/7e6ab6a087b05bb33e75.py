"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Compute priority scores for each activity using a composite metric:
    - Backward critical path length (longest path to project end)
    - Resource scarcity (demand/capacity ratio summed across resources)
    - Downstream fan-out weighted by downstream criticality
    
    Higher priority = scheduled earlier in serial SGS.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors
    predecessors = instance.get("predecessors", [])
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Backward pass: compute longest path from each activity to project end
    # Use topological order (reverse)
    in_degree = pred_count[:]
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use durations
        return durations.copy()
    
    # Backward pass: dist_to_end[i] = durations[i] + max(dist_to_end[succ] for succ in successors[i])
    dist_to_end = np.zeros(n)
    # Process in reverse topological order
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for succ in successors[node]:
            if dist_to_end[succ] > max_succ:
                max_succ = dist_to_end[succ]
        dist_to_end[node] = durations[node] + max_succ
    
    # Resource scarcity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        # demands shape: (n, num_resources) or similar
        # capacities shape: (num_resources,)
        if demands.ndim == 1:
            # Single resource or per-activity demand
            if capacities.ndim == 0:
                cap = float(capacities)
                if cap > 0:
                    resource_score = demands / cap
                else:
                    resource_score = np.zeros(n)
            else:
                # demands is per-activity, capacities is per-resource? Mismatch.
                # Assume single resource
                cap = float(np.max(capacities)) if len(capacities) > 0 else 1.0
                if cap > 0:
                    resource_score = demands / cap
                else:
                    resource_score = np.zeros(n)
        else:
            # demands is (n, R), capacities is (R,)
            # Normalize capacities to avoid division by zero
            safe_caps = np.where(capacities > 0, capacities, 1.0)
            # For each activity, sum of demand/capacity across resources
            # Only count resources where demand > 0
            ratio = demands / safe_caps[np.newaxis, :]
            resource_score = np.sum(ratio, axis=1)
    
    # Downstream fan-out: count of downstream activities weighted by their dist_to_end
    # Use a simple BFS/DFS to count all descendants
    downstream_weight = np.zeros(n)
    # Process in reverse topological order
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        # Sum of (1 + downstream_weight[succ]) for all successors
        # Actually, we want total number of downstream activities
        # Let's compute: num_downstream[node] = sum(1 + num_downstream[succ] for succ in successors[node])
        # But this can double-count. Use a proper count.
        pass
    
    # Simpler: compute for each node, the total "criticality mass" of its descendants
    # descendant_criticality[node] = sum of dist_to_end[succ] for all direct successors
    # + sum of descendant_criticality[succ] for all direct successors
    descendant_crit = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total = 0.0
        for succ in successors[node]:
            total += dist_to_end[succ] + descendant_crit[succ]
        descendant_crit[node] = total
    
    # Normalize components to [0, 1] for combining
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_dist = normalize(dist_to_end)
    norm_res = normalize(resource_score)
    norm_desc = normalize(descendant_crit)
    
    # Combine: weighted sum
    # Emphasize critical path and resource scarcity
    priority_scores = 0.4 * norm_dist + 0.35 * norm_res + 0.25 * norm_desc
    
    # Add a small component for duration itself to break ties
    norm_dur = normalize(durations)
    priority_scores += 0.05 * norm_dur
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin before 120s limit
        return priority_scores
    
    return priority_scores
# EVOLVE-BLOCK-END
