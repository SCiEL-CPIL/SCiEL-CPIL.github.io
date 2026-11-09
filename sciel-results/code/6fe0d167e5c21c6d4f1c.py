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
    3. Compute a "resource tightness" metric based on how much of the resource
       each activity consumes relative to total capacity.
    4. Combine them with a multiplicative boost for high-criticality, high-demand tasks.
    """
    start_time = time.time()
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    
    # Build successor list and in-degrees for topological sort
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
        
    # Check for cycles (should not happen in valid RCPSP, but safety)
    if len(topo_order) != n:
        # Fallback to duration if graph is invalid
        return durations.copy()
    
    # Compute CPL (Critical Path Length) in reverse topological order
    cp_length = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cp_length[u] = durations[u]
        else:
            cp_length[u] = durations[u] + max(cp_length[v] for v in successors[u])
    
    # Check deadline
    if time.time() - start_time > 140:
        return cp_length.copy()
    
    # Compute Resource Demand Intensity
    # max(demand / capacity) across resources
    demand_ratio = np.zeros(n, dtype=float)
    total_demand_weighted = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(len(capacities)):
            if capacities[j] > 0:
                ratio = demands[i][j] / capacities[j]
                demand_ratio[i] = max(demand_ratio[i], ratio)
                # Weighted by duration: longer tasks consuming scarce resources are more critical
                total_demand_weighted[i] += ratio * durations[i]
    
    # Normalize components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp_length)
    ratio_norm = normalize(demand_ratio)
    dur_norm = normalize(durations)
    weighted_demand_norm = normalize(total_demand_weighted)
    
    # Check deadline
    if time.time() - start_time > 145:
        return cp_norm + 0.5 * ratio_norm
    
    # Combined Score
    # Strategy: Strong CPL base, with multiplicative resource boost
    # - High CPL tasks get priority
    # - Among similar CPL, high resource intensity tasks get boosted
    # - Duration-weighted demand captures "long tasks that hog resources"
    # 
    # The multiplicative form ensures resource boost only applies to critical-path tasks
    # We use a stronger resource weight than before to better handle resource contention
    
    scores = cp_norm * (1.0 + 0.8 * ratio_norm + 0.3 * weighted_demand_norm) + 0.15 * dur_norm
    
    return scores
# EVOLVE-BLOCK-END
