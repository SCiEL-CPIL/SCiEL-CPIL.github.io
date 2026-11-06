"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Returns priority scores for tasks based on Weighted Precedence (Sum of Successor Times).
    Tasks with higher scores are selected first by the greedy balancer.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Build successor adjacency list and in-degrees for topological sort
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Kahn's Algorithm for Topological Sorting
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                
    # If there's a cycle, topo_order will be shorter than n.
    # For safety, if not all nodes are processed, we fall back to a simpler heuristic
    # or just use the processed ones and append the rest arbitrarily.
    # However, ALB instances are typically DAGs.
    if len(topo_order) < n:
        # Fallback: Just return times (LPT) if graph is invalid/cyclic
        return times

    # Calculate Weighted Precedence: Sum of times of task and all transitive successors
    weights = np.zeros(n, dtype=float)
    
    # Process in reverse topological order so that successors are already computed
    for u in reversed(topo_order):
        w = times[u]
        for v in successors[u]:
            w += weights[v]
        weights[u] = w
        
    return weights
# EVOLVE-BLOCK-END
