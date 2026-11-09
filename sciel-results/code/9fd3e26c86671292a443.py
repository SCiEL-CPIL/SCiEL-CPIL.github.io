import numpy as np
import time
from collections import deque
import math

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.
    
    Strategy:
    1. Compute Smooth Critical Path Length (S-CPL) using Log-Sum-Exp (Soft-Max).
       This provides a differentiable/smooth approximation of the critical path,
       potentially capturing 'quasi-critical' paths better than hard max.
    2. Compute Resource Demand Intensity (sum of demand/capacity ratios).
    3. Combine them with a multiplicative boost: S-CPL * (1 + alpha * Resource_Intensity).
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

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
        
    if len(topo_order) != n:
        return durations.copy()
    
    # Compute Smooth Critical Path Length (S-CPL) in reverse topological order
    # Using Log-Sum-Exp for soft-max: max(x) ~ T * log(sum(exp(x/T)))
    # We use T=1.0 for a moderate smoothing effect.
    T = 1.0
    s_cp_length = np.zeros(n, dtype=float)
    
    for u in reversed(topo_order):
        if not successors[u]:
            s_cp_length[u] = durations[u]
        else:
            # Get successor CPLs
            succ_vals = [s_cp_length[v] for v in successors[u]]
            # Stable Log-Sum-Exp
            max_val = max(succ_vals)
            # If max_val is very large, exp(succ/T) might overflow.
            # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
            # So T * logsumexp(succ/T) = T * (max(succ)/T + log(sum(exp((succ - max(succ))/T))))
            # = max(succ) + T * log(sum(exp((succ - max(succ))/T)))
            
            exp_terms = [math.exp((val - max_val) / T) for val in succ_vals]
            sum_exp = sum(exp_terms)
            log_sum_exp = max_val + T * math.log(sum_exp)
            
            s_cp_length[u] = durations[u] + log_sum_exp

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return s_cp_length.copy()

    # Compute Resource Demand Intensity
    # Sum of (demand / capacity) across all resources
    # This captures total relative resource consumption
    res_int = np.zeros(n, dtype=float)
    for i in range(n):
        total_ratio = 0.0
        for j in range(len(capacities)):
            if capacities[j] > 0:
                total_ratio += demands[i][j] / capacities[j]
        res_int[i] = total_ratio

    # Normalize components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(s_cp_length)
    res_norm = normalize(res_int)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cp_norm + res_norm

    # Combined Score
    # Multiplicative boost: High CPL tasks get boosted by their resource usage.
    # This prioritizes critical tasks that are also resource-heavy, ensuring they
    # are scheduled before they can block other critical paths.
    # We use a coefficient of 0.5 as in the previous version, which performed well.
    scores = cp_norm * (1.0 + 0.5 * res_norm)
    
    # Add a small duration component to break ties? 
    # The soft-max CPL already incorporates duration.
    # Let's stick to the multiplicative form for simplicity and focus on the CPL improvement.
    
    return scores
