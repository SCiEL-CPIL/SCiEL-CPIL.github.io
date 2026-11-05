"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_priority(instance):
    """
    Compute a priority score for each task using Critical Path Density.
    
    The idea: prioritize tasks that are both:
    1. Bottlenecks on the critical path (high ratio of own time to downstream path length)
    2. High-fanout branching points (many immediate successors)
    
    This should reduce downstream fragmentation and station count.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    
    # Build adjacency: successors[i] = list of tasks that must come after i
    # predecessors[i] = list of tasks that must come before i
    preds = instance.get("predecessors", None)
    
    if preds is not None:
        # preds[i] is the list of predecessor task indices for task i
        successors = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                successors[p].append(i)
    else:
        # No predecessors info, treat as no dependencies
        successors = [[] for _ in range(n)]
    
    # Compute longest path from each node to any sink (downstream path length)
    # Using topological order via DFS or Kahn's algorithm
    
    # Compute in-degrees
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for s in successors[i]:
            in_degree[s] += 1
    
    # Topological sort (Kahn's)
    from collections import deque
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
    
    # If topo_order doesn't contain all nodes, we have a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use task times as priority
        return times
    
    # Compute longest downstream path (including own time) for each node
    # downstream[i] = times[i] + max(downstream[j] for j in successors[i])
    downstream = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if downstream[s] > max_succ:
                max_succ = downstream[s]
        downstream[i] = times[i] + max_succ
    
    # Compute longest upstream path (including own time) for each node
    # upstream[i] = times[i] + max(upstream[j] for j in predecessors[i])
    upstream = np.zeros(n, dtype=float)
    # Build predecessor list
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        if preds is not None:
            for p in preds[i]:
                pred_list[i].append(p)
    
    for i in topo_order:
        max_pred = 0.0
        for p in pred_list[i]:
            if upstream[p] > max_pred:
                max_pred = upstream[p]
        upstream[i] = times[i] + max_pred
    
    # Critical Path Density: ratio of task time to total downstream path
    # Higher ratio means the task is a bigger bottleneck relative to its downstream
    # Avoid division by zero
    eps = 1e-9
    cpd = times / (downstream + eps)
    
    # Count of immediate successors (fanout)
    fanout = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine: weight by CPD and fanout
    # Tasks with high CPD and high fanout should be prioritized
    # Use a combination: cpd * (1 + alpha * fanout)
    alpha = 0.3
    priority_scores = cpd * (1.0 + alpha * fanout)
    
    # Also add a small bonus for tasks with high upstream path (they're on critical path early)
    # Normalize upstream
    max_upstream = np.max(upstream) if np.max(upstream) > 0 else 1.0
    upstream_norm = upstream / max_upstream
    
    # Final priority: blend of CPD-fanout and upstream position
    beta = 0.2
    priority_scores = (1.0 - beta) * priority_scores + beta * upstream_norm
    
    return priority_scores


def priority(instance):
    """
    Score each assembly task; stations are filled with the highest-priority 
    eligible task that fits the cycle time.
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
    except Exception:
        # Fallback to simple longest-task-first
        scores = np.asarray(instance["times"], dtype=float)
    
    # Deadline guard: if we're running too long, just return what we have
    elapsed = time.time() - start_time
    if elapsed > 10:
        # Shouldn't happen for 30 tasks, but safety
        pass
    
    return scores
# EVOLVE-BLOCK-END
