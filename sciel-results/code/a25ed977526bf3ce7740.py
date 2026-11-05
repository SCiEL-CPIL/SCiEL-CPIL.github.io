"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic components:
    1. Bottom Level (BL): Longest path from node to end. (Critical Path)
    2. Resource Pressure (RP): Sum of (demand / capacity). (Resource Contention)
    3. Max Resource Pressure (MRP): Max of (demand / capacity). (Bottleneck Resource)
    4. Duration: Longer jobs get slight priority to reduce fragmentation.
    
    Score = w1*BL + w2*RP + w3*MRP + w4*Dur
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Compute Bottom Level (Longest path to end)
    # Normalize predecessors
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
                
    # If cycle exists (shouldn't in valid instance), fallback to index order
    if len(top_order) < n:
        top_order = list(range(n))
        # This is a safeguard, but BL calc will be wrong if cycle exists.
        # Assuming valid DAG.

    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate

    # 2. Compute Resource Metrics
    if demands is None or capacities is None:
        # Fallback: Just BL
        bl_max = np.max(bottom_level)
        if bl_max > 0:
            return bottom_level / bl_max
        return bottom_level

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

    # 3. Normalize and Combine
    # Use max normalization to preserve relative importance of "most critical" items
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    mrp_max = np.max(max_resource_pressure) if np.max(max_resource_pressure) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0

    bl_norm = bottom_level / bl_max
    rp_norm = resource_pressure / rp_max
    mrp_norm = max_resource_pressure / mrp_max
    dur_norm = durations / dur_max

    # Weights
    # BL is the primary driver (Critical Path)
    # RP and MRP handle resource constraints
    # Dur is a minor tie-breaker (Longer jobs preferred to reduce tail)
    w_bl = 1.0
    w_rp = 0.4
    w_mrp = 0.2
    w_dur = 0.1

    scores = w_bl * bl_norm + w_rp * rp_norm + w_mrp * mrp_norm + w_dur * dur_norm

    return scores
# EVOLVE-BLOCK-END
