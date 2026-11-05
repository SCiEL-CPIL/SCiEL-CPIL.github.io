"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    # predecessors is likely a list of lists, where predecessors[i] contains indices of tasks that must be done before i
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Calculate priority based on Total Downstream Work (Positional Weight)
    # This sums the times of all successors (transitively), rather than taking the max path.
    # This approach prioritizes tasks that unlock a large amount of total work,
    # which can be more effective than critical path in wide networks.
    
    memo = {}
    
    def get_priority(i):
        if i in memo:
            return memo[i]
        
        sum_succ_priority = 0.0
        for s in successors[i]:
            p = get_priority(s)
            sum_succ_priority += p
        
        res = times[i] + sum_succ_priority
        memo[i] = res
        return res

    priorities = np.zeros(n, dtype=float)
    for i in range(n):
        priorities[i] = get_priority(i)
        
    return priorities
# EVOLVE-BLOCK-END
