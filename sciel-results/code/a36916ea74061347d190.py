"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.
    
    Strategy:
    1. Compute Critical Path Length (CPL) from each activity to the project end.
    2. Compute Resource Demand Intensity (max demand/capacity).
    3. Combine them: Prioritize high CPL activities, with a boost for high resource demand.
    """
    start_time = time.time()
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape: (n, num_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape: (num_resources,)
    
    # Get predecessors
    preds = instance["predecessors"]  # list of lists
    
    # Build successor list and in-degree
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute critical path length (longest path from each activity to end)
    cp_length = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cp_length[u] = durations[u]
        else:
            cp_length[u] = durations[u] + max(cp_length[v] for v in successors[u])
    
    # Compute resource demand intensity: max demand/capacity across resources
    # Use vectorized operation for speed
    demand_ratio = np.zeros(n, dtype=float)
    for j in range(len(capacities)):
        if capacities[j] > 0:
            ratio = demands[:, j] / capacities[j]
            demand_ratio = np.maximum(demand_ratio, ratio)
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp_length)
    ratio_norm = normalize(demand_ratio)
    
    # Weighted combination: prioritize critical path with resource demand boost
    # Higher weight on CPL, moderate weight on resource intensity
    weights = {
        'cp': 0.7,
        'ratio': 0.3
    }
    
    scores = (weights['cp'] * cp_norm + 
              weights['ratio'] * ratio_norm)
    
    # Check deadline
    if time.time() - start_time > 145:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
