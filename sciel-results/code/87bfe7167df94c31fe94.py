"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
    
    # Compute rank: longest path from task to end (including own time)
    # Process in reverse topological order
    rank = np.zeros(n)
    # Topological sort (Kahn's algorithm)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Process in reverse topological order to compute rank
    for u in reversed(topo_order):
        if successors[u]:
            max_succ_rank = max(rank[v] for v in successors[u])
            rank[u] = times[u] + max_succ_rank
        else:
            rank[u] = times[u]
    
    # Count number of successors (direct)
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Also compute total number of transitive successors (downstream tasks)
    transitive_succ = np.zeros(n, dtype=int)
    for u in reversed(topo_order):
        if successors[u]:
            transitive_succ[u] = len(successors[u]) + max(transitive_succ[v] for v in successors[u])
        # else stays 0
    
    transitive_succ_f = transitive_succ.astype(float)
    
    # Combine: rank is the primary signal, add a small bonus for more downstream impact
    # Normalize components
    if rank.max() > rank.min():
        rank_norm = (rank - rank.min()) / (rank.max() - rank.min())
    else:
        rank_norm = np.ones(n)
    
    if transitive_succ_f.max() > transitive_succ_f.min():
        ts_norm = (transitive_succ_f - transitive_succ_f.min()) / (transitive_succ_f.max() - transitive_succ_f.min())
    else:
        ts_norm = np.zeros(n)
    
    # Priority: primarily rank, with a smaller weight on transitive successors
    priority_scores = rank_norm + 0.1 * ts_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
