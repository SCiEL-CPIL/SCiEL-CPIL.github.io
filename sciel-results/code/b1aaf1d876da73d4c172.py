import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of predecessor task indices
    cycle_time = instance["cycle_time"]
    
    # Build successor graph
    successors = defaultdict(list)
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Compute longest path to sink (reverse topological order)
    rank = np.zeros(n)
    for node in reversed(topo_order):
        max_succ_rank = 0.0
        for succ in successors[node]:
            if rank[succ] > max_succ_rank:
                max_succ_rank = rank[succ]
        rank[node] = times[node] + max_succ_rank
    
    return rank
