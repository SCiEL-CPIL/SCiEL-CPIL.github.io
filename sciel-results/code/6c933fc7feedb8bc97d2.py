"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 110.0  # Safety margin before 120s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: 
    1. Compute Head (longest path to node) and Tail (longest path from node).
    2. Compute Total Float (Slack) = Critical Path Length - Head - Tail + Duration.
    3. Add a resource-contention component based on demand-to-capacity ratios.
    4. Weights tuned to emphasize critical path structure with resource awareness.
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
    # Fallback to simple duration priority if cycle detected (shouldn't happen in valid RCPSP).
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
    # slack[i] = LS[i] - ES[i]
    # ES[i] = head[i] - durations[i]
    # LF[i] = makespan - (tail[i] - durations[i])
    # LS[i] = LF[i] - durations[i] = makespan - tail[i]
    # slack[i] = (makespan - tail[i]) - (head[i] - durations[i])
    # slack[i] = makespan - tail[i] - head[i] + durations[i]
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - tail[i] - head[i] + durations[i]
    
    # Resource contention: for each activity, compute max demand/capacity ratio
    # Higher ratio means the activity is more likely to cause resource bottlenecks
    n_resources = len(capacities)
    if n_resources > 0 and demands.shape[0] == n and demands.shape[1] == n_resources:
        # Avoid division by zero
        safe_caps = np.where(capacities > 1e-10, capacities, 1.0)
        ratios = demands / safe_caps  # shape: (n, n_resources)
        # Use max ratio across resources as contention score
        contention = np.max(ratios, axis=1)
    else:
        contention = np.ones(n, dtype=float) * 0.5
    
    # Normalize components to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_tail = normalize(tail)
    norm_slack_inv = normalize(-slack)  # Lower slack = Higher priority
    norm_contention = normalize(contention)
    
    # Combine:
    # Emphasize Tail (critical path length), Inverse Slack (urgency), and resource contention.
    # Weights: 0.5 for Tail, 0.3 for Slack, 0.2 for Contention.
    priority_scores = 0.5 * norm_tail + 0.3 * norm_slack_inv + 0.2 * norm_contention
    
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
            # Should not happen for 30 activities, but guard anyway
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
