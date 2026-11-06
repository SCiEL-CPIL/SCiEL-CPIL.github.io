import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    
    # Handle both possible key names for capacities
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.asarray(instance["capancies"], dtype=float)
    
    demands = np.asarray(instance["demands"], dtype=float)

    # 1. Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]

    while current_queue:
        node = current_queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)

    if len(topo_order) != n:
        # Cycle detected or something wrong; fallback to durations
        return durations.copy()

    # 3. Forward Pass: Compute Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]

    project_duration = np.max(EF)

    # 4. Backward Pass: Compute Latest Finish (LF) and Latest Start (LS)
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)

    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        LS[i] = LF[i] - durations[i]

    # 5. Compute Slack (Total Float)
    slack = LS - ES

    # 6. Compute Resource Intensity (RI)
    # RI_i = sum(demand_ij / capacity_j)
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    resource_usage = np.sum(demands / safe_caps[np.newaxis, :], axis=1)

    # 7. Compute Normalized Metrics
    # Normalize slack: lower slack is more critical -> higher priority
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    slack_range = max_slack - min_slack
    if slack_range > 1e-10:
        slack_norm = (max_slack - slack) / slack_range  # Invert: low slack -> high value
    else:
        slack_norm = np.ones(n)

    # Normalize resource usage
    min_ru = np.min(resource_usage)
    max_ru = np.max(resource_usage)
    ru_range = max_ru - min_ru
    if ru_range > 1e-10:
        resource_usage_norm = (resource_usage - min_ru) / ru_range
    else:
        resource_usage_norm = np.zeros(n)

    # 8. Compute Duration-based Priority
    # Longer durations on critical paths are important.
    # We can use duration normalized, or simply use it as a tie-breaker.
    min_dur = np.min(durations)
    max_dur = np.max(durations)
    dur_range = max_dur - min_dur
    if dur_range > 1e-10:
        duration_norm = (durations - min_dur) / dur_range
    else:
        duration_norm = np.zeros(n)

    # 9. Combine Scores
    # Strategy:
    # Primary: Criticality (1 - normalized slack). This is the standard "Latest Finish Time" heuristic.
    # Secondary: Resource Intensity. Bottleneck resources should be scheduled early to avoid blocking.
    # Tertiary: Duration. If criticality is equal, longer tasks might be better to start earlier (LPT)
    #           or shorter (SPT) depending on context. In RCPSP, LPT is often better for makespan
    #           when resources are scarce, as it clears the "big" jobs.
    
    # Let's try a weighted sum.
    # Weights: 0.7 for Criticality, 0.2 for Resource Intensity, 0.1 for Duration (LPT)
    weights = [0.7, 0.2, 0.1]
    
    scores = (weights[0] * slack_norm + 
              weights[1] * resource_usage_norm + 
              weights[2] * duration_norm)
    
    # Alternative: Multiplicative approach might be more sensitive to extremes
    # scores = slack_norm * (1 + 0.5 * resource_usage_norm) * (1 + 0.2 * duration_norm)
    
    # Let's stick to the additive weighted sum for stability, but tune weights.
    # The "Urgency" in the previous code was basically (MaxLF - LF), which is proportional to slack.
    # The previous code had 0.8 urgency + 0.2 RI.
    # Let's try 0.6 Urgency + 0.3 RI + 0.1 Duration.
    
    # Actually, let's look at the "inspiration" scores.
    # Inspiration 3 (1.0241) was "optimizing critical path calculation".
    # Inspiration 2 (1.0347) was "resource pressure".
    # Current (1.0456) is 0.8 Urgency + 0.2 RI.
    
    # Let's try emphasizing Resource Intensity more, as bottleneck resources are key.
    # And maybe less weight on pure duration, as slack already captures criticality.
    
    scores = 0.6 * slack_norm + 0.3 * resource_usage_norm + 0.1 * duration_norm
    
    # Ensure scores are unique if possible to avoid tie-breaking issues in the scheduler
    # (though the scheduler likely handles ties, adding a tiny epsilon based on index can help)
    # But we shouldn't distort the scores too much.
    
    return scores
