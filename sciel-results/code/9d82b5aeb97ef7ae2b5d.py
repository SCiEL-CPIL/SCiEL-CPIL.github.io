"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute priority scores for each activity based on the longest path from
    that activity to the project end (critical path length including its own duration).
    Higher priority = longer remaining path to end.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute longest path from each node to end (reverse topological order)
    # We need to process nodes in reverse topological order
    # First, get topological order
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
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
    
    # If topo_order length != n, there's a cycle; fallback to duration
    if len(topo_order) != n:
        return durations.copy()
    
    # Longest path from node to end (including node's own duration)
    # Process in reverse topological order
    longest_path = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            longest_path[node] = durations[node]
        else:
            max_succ_path = max(longest_path[s] for s in successors[node])
            longest_path[node] = durations[node] + max_succ_path
    
    return longest_path
# EVOLVE-BLOCK-END
