"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Critical Path Density.
    
    Density is defined as task_time / total_critical_path_length_including_task.
    This identifies bottlenecks where the task itself constitutes a significant
    portion of the critical path. This is combined with the number of immediate
    successors to prioritize high-fanout bottlenecks, which helps reduce
    downstream fragmentation and station count.
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
    
    # Compute Critical Path Density: times[i] / longest_downstream[i]
    density = np.zeros(n, dtype=float)
    for i in range(n):
        if longest_downstream[i] > 0:
            density[i] = times[i] / longest_downstream[i]
        else:
            density[i] = 0.0
            
    # Compute immediate successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine Density and Successor Count
    # Prioritize high density tasks that also have high fanout.
    priority_scores = density * (1.0 + succ_count)
    
    return priority_scores
# EVOLVE-BLOCK-END
