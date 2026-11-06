import numpy as np
import time as time_module


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]  # list of lists: predecessors[i] = list of task indices that must come before i
    n = len(times)
    
    # Build successor graph
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute critical path: longest path from task i to any sink
    # critical[i] = times[i] + max(critical[j] for j in successors[i]), base case: times[i] if no successors
    # Use topological order (reverse)
    # First, compute in-degrees and do topological sort
    in_degree = [len(predecessors[i]) for i in range(n)]
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
    
    # Process in reverse topological order to compute critical paths
    critical = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if critical[s] > max_succ:
                max_succ = critical[s]
        critical[i] = times[i] + max_succ
    
    # Also compute number of successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Composite priority: emphasize critical path, then time, then successors
    # This should help the greedy algorithm make better choices
    priority_scores = critical * 1.0 + times * 0.1 + num_successors * 0.01
    
    return priority_scores
