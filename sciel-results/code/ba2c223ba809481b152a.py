"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    
    Uses a combination of:
    - Longest remaining path (backward pass) as primary priority
    - High resource demand as secondary tie-breaker
    - Short duration as tertiary tie-breaker (to reduce fragmentation)
    
    The idea: activities that are on the critical path (longest remaining path)
    should be scheduled first. Among those, activities with high resource demand
    should go first to avoid resource contention later.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
    
    # Compute longest path from each activity to the end (backward pass)
    # longest[i] = max duration of any path starting at activity i (including i's duration)
    longest = np.zeros(n, dtype=float)
    # Topological order (reverse of topological sort of predecessors)
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                in_degree[j] += 1
    
    # Get topological order using Kahn's algorithm
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't include all nodes, add remaining (handle cycles gracefully)
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Backward pass: compute longest remaining path
    # Process in reverse topological order
    for node in reversed(topo_order):
        if successors[node]:
            longest[node] = durations[node] + max(longest[s] for s in successors[node])
        else:
            longest[node] = durations[node]
    
    # Compute total resource demand per activity (sum across all resources)
    demands = instance["demands"]
    # demands is likely a list/array of shape (n, num_resources) or similar
    total_demand = np.sum(np.asarray(demands, dtype=float), axis=1)
    
    # Normalize components to [0, 1] range for combining
    max_longest = np.max(longest) if np.max(longest) > 0 else 1.0
    max_demand = np.max(total_demand) if np.max(total_demand) > 0 else 1.0
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    
    # Priority score: higher is better for scheduling earlier
    # Primary: longest remaining path (critical path activities first)
    # Secondary: high resource demand (schedule resource-heavy tasks early)
    # Tertiary: shorter duration (to reduce idle time / fragmentation)
    
    norm_longest = longest / max_longest
    norm_demand = total_demand / max_demand
    norm_dur = durations / max_dur  # shorter duration = lower value, we want to prefer shorter
    
    # Combine: weighted sum
    # Weights: 0.6 for longest path, 0.3 for demand, 0.1 for (1 - duration)
    score = 0.6 * norm_longest + 0.3 * norm_demand + 0.1 * (1.0 - norm_dur)
    
    return score
# EVOLVE-BLOCK-END
