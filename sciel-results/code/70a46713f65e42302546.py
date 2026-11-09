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
    3. Combine them: Prioritize high CPL activities, with a boost for high resource intensity.
    4. Use Duration as a minor tie-breaker.
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
    # Smooth surrogate for max(demand / capacity) via log-sum-exp
    n_res = len(capacities)
    demand_ratio = np.zeros(n, dtype=float)
    for j in range(n_res):
        if capacities[j] > 0:
            ratio_col = demands[:, j] / capacities[j]
        else:
            ratio_col = np.zeros(n, dtype=float)
        # Clip to avoid log(0) issues; use a small floor
        ratio_clipped = np.maximum(ratio_col, 1e-10)
        # Log-sum-exp: log(sum(exp(x))) = max_x + log(sum(exp(x - max_x)))
        max_val = np.max(ratio_clipped)
        if max_val < 1e-10:
            demand_ratio += 0.0
        else:
            # Use temperature to control smoothness; lower = sharper max
            temp = 0.5
            shifted = ratio_clipped / temp
            shifted_max = np.max(shifted)
            lse = temp * (shifted_max + np.log(np.sum(np.exp(shifted - shifted_max))))
            demand_ratio += lse
    
    # Compute "Resource Tail": longest path weighted by resource intensity
    # This captures how much resource-heavy work lies ahead of each activity
    res_tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        # Base: own resource intensity * duration
        own_weight = demand_ratio[u] * durations[u]
        if not successors[u]:
            res_tail[u] = own_weight
        else:
            # Propagate: own weight + max of successors' tails
            max_succ = max(res_tail[v] for v in successors[u])
            res_tail[u] = own_weight + max_succ
    
    # Normalize components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(cp_length)
    ratio_norm = normalize(demand_ratio)
    tail_norm = normalize(res_tail)
    dur_norm = normalize(durations)
    
    # Check deadline
    if time.time() - start_time > 145:
        return cp_norm + 0.3 * tail_norm
    
    # Combined Score
    # Primary: CPL (critical path length)
    # Secondary: Resource tail (downstream resource pressure)
    # Tertiary: Own resource intensity (for immediate scheduling decisions)
    # Minor: Duration as tie-breaker
    #
    # The resource tail captures the idea that an activity with many
    # resource-intensive successors should be prioritized to avoid
    # delaying those critical downstream tasks.
    
    scores = cp_norm * (1.0 + 0.3 * ratio_norm) + 0.25 * tail_norm + 0.05 * dur_norm
    
    return scores
# EVOLVE-BLOCK-END
