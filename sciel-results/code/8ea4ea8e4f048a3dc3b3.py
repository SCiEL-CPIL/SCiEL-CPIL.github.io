"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity using 'resource bottleneck centrality':
    For each activity i, compute:
      - downstream_count[i]: number of activities that transitively depend on i
      - bottleneck_score[i]: sum over resources r of (demand[i][r] / capacity[r]) * downstream_count[i]
    Then combine with a critical-path-like term (longest path from i to end) to balance
    urgency and gating effect.
    
    Final priority = alpha * normalized_downstream_weighted_bottleneck + beta * normalized_critical_path
    We use alpha=0.6, beta=0.4 as a structural choice.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors
    predecessors = instance.get("predecessors", [])
    # predecessors[i] is a list of indices of activities that must finish before i starts
    successors = [[] for _ in range(n)]
    for i in range(n):
        if i < len(predecessors):
            for pred in predecessors[i]:
                successors[pred].append(i)
    
    # Compute downstream count (number of transitive successors) for each activity
    # Use reverse topological order
    # First, compute in-degrees and do a topological sort
    in_degree = [0] * n
    for i in range(n):
        if i < len(predecessors):
            for pred in predecessors[i]:
                in_degree[i] += 1  # i depends on pred, so i's in-degree increases
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_copy[succ] -= 1
            if in_degree_copy[succ] == 0:
                queue.append(succ)
    
    # If not all nodes are in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use duration-based priority
        return durations.astype(float)
    
    # Compute downstream count in reverse topological order
    downstream_count = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        # downstream_count[node] = 1 + sum of downstream_count of direct successors
        # But we want the count of all transitive successors, not including self
        # Actually, let's compute it as: number of nodes that can reach the end through node
        # Simpler: downstream_count[i] = sum over direct successors j of (1 + downstream_count[j])
        # Wait, that would double-count. Let me think again.
        # 
        # Actually, for "number of downstream activities", we want the size of the transitive closure
        # of successors. We can compute this with a BFS/DFS from each node, but that's O(n^2).
        # For n=30, that's fine.
        pass
    
    # For n=30, let's just do BFS from each node to count transitive successors
    downstream_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = list(successors[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(successors[node])
        downstream_count[i] = len(visited)
    
    # Compute critical path length from each node to the end (longest path from node to any sink)
    # Process in reverse topological order
    critical_path = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if successors[node]:
            critical_path[node] = durations[node] + np.max([critical_path[s] for s in successors[node]])
        else:
            critical_path[node] = durations[node]
    
    # Compute resource bottleneck score
    # demands[i][r] is the demand of activity i on resource r
    # capacities[r] is the capacity of resource r
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Handle potential zero capacities
    safe_capacities = np.where(capacities > 0, capacities, 1.0)
    
    # For each activity i, bottleneck_score[i] = sum_r (demands[i][r] / safe_capacities[r]) * downstream_count[i]
    # demands shape: (n, num_resources)
    # safe_capacities shape: (num_resources,)
    
    if demands.ndim == 1:
        # Single resource or flat array? Assume it's (n,) for single resource
        num_resources = 1
        demands_2d = demands.reshape(-1, 1)
    else:
        demands_2d = demands
        num_resources = demands_2d.shape[1]
    
    # Compute per-activity resource usage ratio
    # resource_ratio[i] = sum_r (demands[i][r] / safe_capacities[r])
    resource_ratio = np.sum(demands_2d / safe_capacities, axis=1)
    
    # bottleneck_score[i] = resource_ratio[i] * downstream_count[i]
    bottleneck_score = resource_ratio * downstream_count
    
    # Normalize both scores to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    normalized_bottleneck = normalize(bottleneck_score)
    normalized_critical = normalize(critical_path)
    
    # Combine
    alpha = 0.6
    beta = 0.4
    priority_scores = alpha * normalized_bottleneck + beta * normalized_critical
    
    return priority_scores
# EVOLVE-BLOCK-END
