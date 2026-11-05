"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each assembly task using a composite heuristic:
    - Downstream workload (sum of all successor times)
    - Critical path length from task to end
    - Successor count (out-degree)
    - Task duration itself
    
    The priority is designed to place tasks that are bottlenecks for many
    downstream tasks early, while also considering the critical path.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    
    # Build successor graph
    predecessors = instance.get("predecessors", [list() for _ in range(n)])
    
    # Ensure predecessors is a list of lists
    if isinstance(predecessors, (list, tuple)):
        pred_list = [list(p) for p in predecessors]
    else:
        pred_list = [list(p) for p in predecessors]
    
    # Build successor lists
    successors = defaultdict(list)
    in_degree = [0] * n
    for j in range(n):
        for i in pred_list[j]:
            successors[i].append(j)
            in_degree[j] += 1
    
    # Compute longest path from each task to any terminal (downstream critical path)
    # Use topological order
    in_deg = in_degree[:]
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    
    # Longest path from u to any sink (reverse topological)
    longest_to_sink = [0.0] * n
    for u in reversed(topo_order):
        if successors[u]:
            longest_to_sink[u] = max(longest_to_sink[u], max(longest_to_sink[v] for v in successors[u]))
        longest_to_sink[u] += times[u]
    
    # Sum of all successor times (total downstream workload)
    downstream_sum = [0.0] * n
    for u in reversed(topo_order):
        total = 0.0
        for v in successors[u]:
            total += times[v] + downstream_sum[v]
        downstream_sum[u] = total
    
    # Out-degree (number of immediate successors)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Composite priority:
    # - Higher downstream_sum means this task gates a lot of work
    # - Higher longest_to_sink means it's on a critical path
    # - Higher out_degree means it's a branching point
    # - Critical ratio: downstream workload relative to task duration
    #   (high ratio = bottleneck that causes idle time in successors)
    
    # Normalize components to [0, 1] to avoid scale issues
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_downstream = normalize(downstream_sum)
    norm_critical = normalize(longest_to_sink)
    norm_outdeg = normalize(out_degree)
    norm_time = normalize(times)
    
    # Critical ratio: downstream_sum / (task_time + epsilon)
    # High ratio means this task is a bottleneck relative to its duration
    eps = 1e-6
    critical_ratio = downstream_sum / (times + eps)
    norm_crit_ratio = normalize(critical_ratio)
    
    # Weights: emphasize downstream workload, critical ratio, and critical path
    w_downstream = 0.3
    w_critical = 0.25
    w_crit_ratio = 0.25
    w_outdeg = 0.1
    w_time = 0.1
    
    scores = (w_downstream * norm_downstream + 
              w_critical * norm_critical + 
              w_crit_ratio * norm_crit_ratio +
              w_outdeg * norm_outdeg + 
              w_time * norm_time)
    
    return scores
# EVOLVE-BLOCK-END
