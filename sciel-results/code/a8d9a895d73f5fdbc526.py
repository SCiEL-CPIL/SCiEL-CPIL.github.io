# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Uses a combination of:
    1. Critical path length (longest path from activity to end) - primary driver
    2. Resource demand intensity via soft-max over resource ratios - captures scarcity
    3. Duration as tiebreaker
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors/successors
    preds = instance.get("predecessors", None)
    succs = instance.get("successors", None)
    
    # Build adjacency if not provided
    if preds is None:
        # Try to infer from other keys
        preds = instance.get("preds", None)
    if succs is None:
        succs = instance.get("succs", None)
    
    # Build predecessor list (list of lists)
    if preds is not None:
        pred_list = []
        for i in range(n):
            p = preds[i] if i < len(preds) else []
            if p is None:
                p = []
            pred_list.append(list(p))
    else:
        pred_list = [[] for _ in range(n)]
    
    # Build successor list
    if succs is not None:
        succ_list = []
        for i in range(n):
            s = succs[i] if i < len(succs) else []
            if s is None:
                s = []
            succ_list.append(list(s))
    else:
        succ_list = [[] for _ in range(n)]
        for i in range(n):
            for p in pred_list[i]:
                if p < n:
                    succ_list[p].append(i)
    
    # Compute longest path from each node to any sink (backward pass)
    # This is the "critical path length" from activity i to the end
    # Use topological order (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            if p < n:
                in_degree[i] += 1
    
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_list[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes processed, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes in order
        processed = set(topo_order)
        for i in range(n):
            if i not in processed:
                topo_order.append(i)
    
    # Backward pass: compute longest path from each node to end
    # longest[i] = durations[i] + max(longest[j] for j in successors of i)
    # Process in reverse topological order
    longest = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ = 0.0
        for j in succ_list[i]:
            if j < n:
                max_succ = max(max_succ, longest[j])
        longest[i] = durations[i] + max_succ
    
    # Compute resource demand intensity
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Ensure shapes are correct
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, -1)
        if capacities_arr.ndim == 1:
            capacities_arr = capacities_arr.reshape(1, -1)
        
        # Compute resource ratios: demand / capacity for each resource
        # Shape: (n, R)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = demands_arr / capacities_arr
        ratios = np.nan_to_num(ratios, nan=0.0, posinf=10.0, neginf=0.0)
        
        # Soft-max over resources to capture scarcity intensity
        # Use log-sum-exp for numerical stability
        tau = 0.1
        scaled = ratios / tau
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        # Handle case where all are very negative
        max_scaled = np.where(np.isfinite(max_scaled), max_scaled, 0.0)
        exp_scaled = np.exp(scaled - max_scaled)
        log_sum_exp = max_scaled.squeeze(axis=1) + np.log(np.sum(exp_scaled, axis=1))
        
        # Convert back to a scale comparable to longest path
        resource_score = log_sum_exp * tau
        # Normalize to be in a reasonable range
        if np.max(resource_score) > 0:
            resource_score = resource_score / np.max(resource_score) * np.max(longest)
    else:
        resource_score = np.zeros(n, dtype=float)
    
    # Combine: critical path length is primary, resource scarcity is secondary
    # Weighting: 70% critical path, 30% resource intensity
    # This ensures that activities on the critical path are prioritized,
    # but among similar critical path lengths, those with higher resource
    # demands get priority (they're harder to schedule and block more)
    
    # Normalize longest path to [0, 1]
    max_longest = np.max(longest)
    if max_longest > 0:
        norm_longest = longest / max_longest
    else:
        norm_longest = np.zeros(n, dtype=float)
    
    # Normalize resource score to [0, 1]
    max_res = np.max(resource_score)
    if max_res > 0:
        norm_res = resource_score / max_res
    else:
        norm_res = np.zeros(n, dtype=float)
    
    # Final score
    score = 0.7 * norm_longest + 0.3 * norm_res
    
    # Add small duration tiebreaker
    max_dur = np.max(durations)
    if max_dur > 0:
        score += 0.05 * (durations / max_dur)
    
    return score
# EVOLVE-BLOCK-END
