"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Critical Path Density:
    - Ratio of task duration to its longest downstream path (bottleneck density)
    - Combined with immediate successor count (fanout) to prioritize tasks that are
      both bottlenecks in the critical path and high-fanout branching points.
    
    This approach aims to reduce downstream fragmentation and station count by
    scheduling tasks that constrain the critical path and have many successors first.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor lists from predecessors
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    successors[pred].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i in predecessors:
            for pred in predecessors[i]:
                if pred is not None:
                    in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't include all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
    
    # Compute longest downstream path (including self) in reverse topological order
    longest_downstream = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        succs = successors[node]
        if not succs:
            longest_downstream[node] = times[node]
        else:
            max_succ_path = 0.0
            for s in succs:
                if longest_downstream[s] > max_succ_path:
                    max_succ_path = longest_downstream[s]
            longest_downstream[node] = times[node] + max_succ_path
    
    # Compute Critical Path Density: ratio of task duration to longest downstream path
    # This measures how much of the critical path is consumed by this task itself
    with np.errstate(divide='ignore', invalid='ignore'):
        cpd = np.where(longest_downstream > 0, times / longest_downstream, 0.0)
    
    # Compute immediate successor count (fanout)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Normalize components to [0, 1]
    max_cpd = cpd.max() if n > 0 else 1.0
    max_sc = succ_count.max() if n > 0 else 1.0
    max_ld = longest_downstream.max() if n > 0 else 1.0
    
    if max_cpd > 0:
        norm_cpd = cpd / max_cpd
    else:
        norm_cpd = np.zeros(n, dtype=float)
    
    if max_sc > 0:
        norm_sc = succ_count / max_sc
    else:
        norm_sc = np.zeros(n, dtype=float)
    
    if max_ld > 0:
        norm_ld = longest_downstream / max_ld
    else:
        norm_ld = np.zeros(n, dtype=float)
    
    # Weighted combination:
    # - Critical Path Density (primary): identifies bottleneck tasks on the critical path
    # - Successor count (secondary): identifies high-fanout branching points
    # - Longest downstream path (tertiary): absolute criticality as tiebreaker
    priority_scores = 0.5 * norm_cpd + 0.3 * norm_sc + 0.2 * norm_ld
    
    return priority_scores
# EVOLVE-BLOCK-END
