"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin, problem is small
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    
    # predecessors is likely a list of lists, where predecessors[i] is a list of task indices that must precede i.
    # We need to build the successor graph to compute bottom levels (ranks).
    # succ[i] is a list of tasks that directly follow i.
    succ = [[] for _ in range(n)]
    
    for i in range(n):
        if i < len(predecessors):
            for p in predecessors[i]:
                # p is a predecessor of i, so i is a successor of p
                if 0 <= p < n:
                    succ[p].append(i)
        
    # Check time
    if time.time() > deadline:
        return times

    # Compute Rank (Bottom Level)
    # Rank(i) = times[i] + max(Rank(j) for j in succ[i])
    # We can use DFS with memoization.
    rank = [-1.0] * n
    
    def get_rank(i):
        if rank[i] >= 0:
            return rank[i]
        max_succ_rank = 0.0
        for j in succ[i]:
            r = get_rank(j)
            if r > max_succ_rank:
                max_succ_rank = r
        rank[i] = times[i] + max_succ_rank
        return rank[i]
        
    for i in range(n):
        get_rank(i)
        
    return np.asarray(rank, dtype=float)
# EVOLVE-BLOCK-END
