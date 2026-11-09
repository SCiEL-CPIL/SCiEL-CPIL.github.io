"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
       This provides a smooth approximation of the critical path, capturing 'quasi-critical' paths.
    2. Compute Resource Demand Intensity (sum of demand/capacity ratios).
    3. Compute Static Slack (Total Float) from CPM: earliest start + duration + CPL - project_length.
       Low slack indicates criticality.
    4. Combine them: Prioritize high S-CPL activities, with a boost for high resource intensity
       and low slack (high criticality).
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
    T = 0.5  # Lower temperature for sharper max approximation while retaining smoothness
    s_cp_length = np.zeros(n, dtype=float)
    
    for u in reversed(topo_order):
        if not successors[u]:
            s_cp_length[u] = durations[u]
        else:
            # Get successor CPLs
            succ_vals = [s_cp_length[v] for v in successors[u]]
            # Stable Log-Sum-Exp
            max_val = max(succ_vals)
            exp_terms = [math.exp((val - max_val) / T) for val in succ_vals]
            sum_exp = sum(exp_terms)
            log_sum_exp = max_val + T * math.log(sum_exp)
            
            s_cp_length[u] = durations[u] + log_sum_exp

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return s_cp_length.copy()

    # Compute Earliest Start times for Slack calculation
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(es[p] + durations[p] for p in preds[u])
    
    # Project length (approximate using max ES + duration)
    project_length = max(es[i] + durations[i] for i in range(n))
    
    # Compute Static Slack (Total Float)
    # Slack = (ES + Duration + CPL) - Project_Length
    # Lower slack means more critical.
    # Note: We use the smooth CPL here for consistency, though hard CPL is standard for slack.
    # Using smooth CPL might give a "smooth slack" which is also informative.
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = (es[i] + durations[i] + s_cp_length[i]) - project_length
    
    # Compute Resource Demand Intensity
    # Use soft-max of demand/capacity ratios to capture multiple near-saturated resources
    # rather than just the single worst one (per cell findings: 1.088 -> 1.096)
    res_int = np.zeros(n, dtype=float)
    T_res = 0.3  # Temperature for resource soft-max
    for i in range(n):
        ratios = []
        for j in range(len(capacities)):
            if capacities[j] > 0:
                ratios.append(demands[i][j] / capacities[j])
        if not ratios:
            res_int[i] = 0.0
        else:
            # Soft-max: T * log(sum(exp(r/T))) approximates max but accounts for all resources
            max_r = max(ratios)
            exp_terms = [math.exp((r - max_r) / T_res) for r in ratios]
            res_int[i] = max_r + T_res * math.log(sum(exp_terms))

    # Normalize components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    cp_norm = normalize(s_cp_length)
    res_norm = normalize(res_int)
    dur_norm = normalize(durations)
    
    # For Slack, we want LOW slack to have HIGH priority.
    # So we invert it: high "criticality" = 1 - normalized_slack
    # But careful: if all slacks are 0, normalize returns 0, so 1-0=1 (all critical).
    # If slacks vary, low slack -> low norm -> high (1-norm).
    slack_norm = normalize(slack)
    crit_norm = 1.0 - slack_norm

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cp_norm + res_norm

    # Combined Score
    # Base: Smooth CPL (primary driver)
    # Boost: Resource Intensity (scarcity)
    # Boost: Criticality (low slack)
    #
    # We use a multiplicative form for the resource boost to ensure it only amplifies critical tasks.
    # We add a small additive term for criticality to break ties among high-CPL tasks.
    #
    # weights:
    #   alpha = 0.5 for resource intensity
    #   beta  = 0.2 for criticality (slack)
    
    # Combined Score
    # Primary: Smooth CPL (critical path dominance)
    # Multiplicative resource boost: amplifies critical tasks that are also resource-heavy
    # Additive criticality: breaks ties among similar-CPL tasks
    # Additive duration: final tiebreaker, longer tasks on critical paths get slight edge
    scores = cp_norm * (1.0 + 0.4 * res_norm) + 0.25 * crit_norm + 0.1 * dur_norm
    
    return scores
# EVOLVE-BLOCK-END
