"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy:
    1. Compute CPM metrics: Head, Tail, Slack.
    2. Compute a resource fit score (1/(1+max_ratio)) to favor activities
       that use resources efficiently relative to capacity.
    3. Combine tail (critical path) and resource fit into a composite score.
    
    The resource fit term uses 1/(1+max_ratio) where max_ratio is the maximum
    demand/capacity ratio across all resources. This properly handles cases
    where resources are nearly exhausted and provides better discrimination
    than linear penalties.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape: (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    if n == 0:
        return np.array([])
    
    # Convert predecessors to a consistent format
    if isinstance(predecessors, np.ndarray):
        pred_lists = []
        for i in range(n):
            pred_lists.append(list(predecessors[i]))
    else:
        pred_lists = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if i not in successors[p]:
                successors[p].append(i)
    
    # Topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't contain all nodes, there's a cycle. 
    # Fallback to simple duration priority if cycle detected.
    if len(topo_order) < n:
        return durations.copy()

    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail) if n > 0 else 0.0
    
    # Total Float / Slack
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
        
    # Resource Fit Score: 1/(1 + max_ratio)
    # Higher fit means the activity uses resources more efficiently
    # (lower maximum demand-to-capacity ratio)
    resource_fit = np.zeros(n, dtype=float)
    n_resources = len(capacities)
    
    for i in range(n):
        max_ratio = 0.0
        for r in range(n_resources):
            if capacities[r] > 0:
                ratio = demands[i][r] / capacities[r]
                if ratio > max_ratio:
                    max_ratio = ratio
            else:
                if demands[i][r] > 0:
                    max_ratio = 100.0  # Very poor fit if capacity is 0
        resource_fit[i] = 1.0 / (1.0 + max_ratio)

    # Normalize components to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_tail = normalize(tail)
    norm_slack_inv = normalize(-slack) # Lower slack = Higher priority
    norm_fit = normalize(resource_fit)  # Higher fit = better
    
    # Combine:
    # Weight tail (critical path urgency) and resource fit (efficiency)
    # The resource fit term properly discriminates between activities
    # that will cause bottlenecks vs those that won't.
    w_tail = 0.50
    w_slack = 0.30
    w_fit = 0.20
    
    priority_scores = (w_tail * norm_tail + 
                       w_slack * norm_slack_inv + 
                       w_fit * norm_fit)
    
    return priority_scores


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE_SECONDS:
            # Fallback
            return np.asarray(instance["durations"], dtype=float)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
