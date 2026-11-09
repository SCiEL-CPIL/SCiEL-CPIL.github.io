"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_SECONDS = 145.0  # safety margin before 150s kill


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: Lexicographic priority using:
    1. Minimum Total Slack (Urgency)
    2. Maximum Tail (Critical Path Length)
    3. Maximum Resource Demand Intensity (Resource Pressure)
    4. Maximum Duration (LPT)
    
    This approach avoids the dilution effect of weighted sums by creating
    a hierarchical scoring system where primary criteria dominate.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]  # shape: (n_activities, n_resources)
    capacities = instance["capacities"]  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    
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
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # head[i] = longest duration from any source to i (including i's own duration)
    head = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            head[i] = durations[i] + max(head[p] for p in pred_lists[i])
        else:
            head[i] = durations[i]
    
    # Project makespan without resource constraints (longest path)
    makespan_ignoring_resources = max(tail[i] for i in range(n)) if n > 0 else 0.0
    
    # Total float / slack
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = makespan_ignoring_resources - head[i] - tail[i] + durations[i]
    
    # Resource demand intensity
    demand_intensity = np.zeros(n, dtype=float)
    n_resources = len(capacities)
    for i in range(n):
        for r in range(n_resources):
            cap = capacities[r]
            if cap > 0:
                demand_intensity[i] += demands[i][r] / cap
            else:
                demand_intensity[i] += demands[i][r] * 10.0
    
    # Smooth scoring using normalized values
    # We want: Min Slack, Max Tail, Max Demand, Max Duration
    # Normalize each to [0, 1] range
    
    def normalize(arr, invert=False):
        """Normalize to [0, 1]. If invert=True, higher original value -> lower normalized score."""
        arr = np.asarray(arr, dtype=float)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-10:
            return np.ones_like(arr) * 0.5
        norm = (arr - min_val) / (max_val - min_val)
        if invert:
            norm = 1.0 - norm
        return norm
    
    # Higher score = higher priority
    # Slack: lower is better -> invert
    norm_slack = normalize(slack, invert=True)
    # Tail: higher is better
    norm_tail = normalize(tail, invert=False)
    # Demand intensity: higher is better
    norm_demand = normalize(demand_intensity, invert=False)
    # Duration: higher is better
    norm_dur = normalize(durations, invert=False)
    
    # Weighted sum with emphasis on slack and tail
    # Based on knowledge: 0.45 slack, 0.22 tail, 0.25 demand, 0.08 duration
    score = 0.45 * norm_slack + 0.22 * norm_tail + 0.25 * norm_demand + 0.08 * norm_dur
    
    return score


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
