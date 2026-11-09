"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute a 'bottleneck density' priority for each task.
    
    The idea:
    - For each task, compute the length of the longest path from that task to any
      sink (i.e., the maximum remaining processing time downstream including itself).
    - Also compute the minimum number of stations required to complete the downstream
      subtree, approximated by ceil(longest_path_downstream / cycle_time) but more
      accurately via a greedy lower bound on station count for the subgraph.
    
    However, a simpler and effective approach that captures the spirit of "bottleneck density":
    - Compute bottom level (longest path from task to any terminal node, including task time).
    - Compute the number of immediate successors.
    - Priority = bottom_level / max(1, min_stations_needed_downstream)
    
    But computing min_stations_needed_downstream exactly is hard. Instead, we use:
    - bottom_level (longest path to sink)
    - A proxy for scarcity: how many tasks are in the downstream cone, or the depth.
    
    A well-known heuristic in assembly line balancing is the "rank" or "bottom level"
    combined with successor count. Let's use:
    
    priority[i] = (bottom_level[i] + epsilon) / (1 + num_successors[i])
    
    This penalizes tasks that have many successors (they unlock many tasks, so placing them
    early is good, but the division by successor count means tasks with few successors and
    high bottom level get higher priority). Wait, that's backwards.
    
    Let me reconsider. The strategy says: "ratio of a task's processing time to the minimum 
    number of remaining stations required to complete its downstream subtree".
    
    So: priority[i] = time[i] / min_stations_downstream[i]
    
    Where min_stations_downstream[i] is an estimate of how many stations are needed to complete
    all tasks that depend on i (including i).
    
    A simple lower bound on stations needed for a subgraph is ceil(total_time_in_subgraph / cycle_time),
    but that's not quite right because of precedence constraints.
    
    Another lower bound: ceil(longest_path_in_subgraph / cycle_time).
    
    Let's use: min_stations_downstream[i] = max(1, ceil(bottom_level[i] / cycle_time))
    
    Then priority[i] = time[i] / min_stations_downstream[i]
    
    This means: if a task has a long downstream path (many stations needed), its time is "spread out"
    over many stations, so it gets lower priority per unit time. If a task is in a tight section
    (few stations needed downstream), its time is concentrated, so it gets higher priority.
    
    This should force the greedy filler to place high-priority tasks (those in tight bottlenecks) first.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            if pred >= 0:
                successors[pred].append(i)
    
    # Compute bottom level (longest path from node to any sink, including node's time)
    # Use topological order (reverse of topological sort of predecessors)
    # Since predecessors[i] gives direct predecessors of i, we can compute in topological order.
    
    # First, compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            if pred >= 0:
                in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # bottom_level[i] = times[i] + max(bottom_level[j] for j in successors[i])
    bottom_level = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(top_order):
        max_succ_bl = 0.0
        for v in successors[u]:
            if bottom_level[v] > max_succ_bl:
                max_succ_bl = bottom_level[v]
        bottom_level[u] = times[u] + max_succ_bl
    
    # Estimate min stations downstream: ceil(bottom_level[i] / cycle_time), at least 1
    min_stations = np.maximum(1.0, np.ceil(bottom_level / cycle_time))
    
    # Priority = time[i] / min_stations[i]
    priority_scores = times / min_stations
    
    return priority_scores
# EVOLVE-BLOCK-END
