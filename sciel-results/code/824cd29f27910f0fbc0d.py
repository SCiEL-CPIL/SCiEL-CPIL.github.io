"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_priority(instance):
    """
    Compute a priority score for each activity to minimize makespan.
    
    Strategy: Use a heuristic that combines:
    1. Longest processing time (LPT) - classical scheduling heuristic
    2. Number of successors (critical path influence)
    3. Resource demand intensity (higher demand = harder to schedule, prioritize)
    4. Position in critical path
    
    The score should be such that higher priority activities are scheduled first.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]  # shape: (n_activities, n_resources)
    capacities = instance["capacities"]  # shape: (n_resources,)
    predecessors = instance["predecessors"]  # list of lists or similar
    
    n = len(durations)
    
    # Convert predecessors to a consistent format
    # predecessors[i] is a list of indices of activities that must finish before i
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
    
    # Compute longest path from each node to any sink (downward critical path)
    # This gives the "tail" length - how much time remains after starting this activity
    # Use topological order
    in_degree = [len(pred_lists[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # For deterministic behavior, process in order
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # tail[i] = longest duration from i to any sink (including i's own duration)
    tail = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo_order):
        if not successors[i]:
            tail[i] = durations[i]
        else:
            tail[i] = durations[i] + max(tail[j] for j in successors[i])
    
    # Compute earliest start times assuming no resource constraints
    # ES[i] = max(ES[p] + duration[p] for p in predecessors[i])
    ES = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            ES[i] = max(ES[p] + durations[p] for p in pred_lists[i])
        else:
            ES[i] = 0.0
    
    # Compute latest start times (LS) from sinks backwards
    # Project makespan without resource constraints = max(ES[i] + durations[i])
    project_end = max(ES[i] + durations[i] for i in range(n))
    
    # LS[i] = project_end - tail[i] (latest start time to finish by project_end)
    LS = np.zeros(n, dtype=float)
    for i in range(n):
        LS[i] = project_end - tail[i]
    
    # Total float / slack = LS[i] - ES[i]
    # Activities with low slack are on or near the critical path and should be prioritized
    slack = LS - ES
    
    # Resource demand intensity: sum of (demand / capacity) for each resource
    # Higher ratio means tighter resource constraint
    demand_intensity = np.zeros(n, dtype=float)
    for i in range(n):
        for r in range(len(capacities)):
            if capacities[r] > 0:
                demand_intensity[i] += demands[i][r] / capacities[r]
            else:
                demand_intensity[i] += demands[i][r] * 10.0
    
    # Combine heuristics:
    # 1. Tail length (critical path) - dominant factor
    # 2. Slack (inverse) - activities with less slack are more critical
    # 3. Resource demand intensity - secondary
    # 4. Duration - tertiary
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    def normalize_inverse(arr):
        """Normalize so that lower values get higher normalized scores"""
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (mx - arr) / (mx - mn)
    
    norm_tail = normalize(tail)
    norm_slack = normalize_inverse(slack)  # Lower slack = higher priority
    norm_demand = normalize(demand_intensity)
    norm_dur = normalize(durations)
    
    # Weighted combination
    # Tail is most important for makespan, then slack, then demand, then duration
    priority_scores = 1.0 * norm_tail + 0.8 * norm_slack + 0.3 * norm_demand + 0.1 * norm_dur
    
    return priority_scores


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled first when eligible).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
        return scores
    except Exception:
        # Fallback: longest processing time
        try:
            return np.asarray(instance["durations"], dtype=float)
        except Exception:
            n = len(instance.get("durations", []))
            return np.ones(n, dtype=float)
# EVOLVE-BLOCK-END
