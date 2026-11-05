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
            
    # Calculate priority based on longest path to sink (downstream weight)
    # We need to process nodes in reverse topological order.
    # We can use Kahn's algorithm on the reversed graph or just a DFS with memoization.
    # Given n is small (30), DFS is fine and simpler to implement robustly without extra arrays for topological sort if we just compute recursively.
    
    memo = {}
    
    def get_priority(i):
        if i in memo:
            return memo[i]
        
        max_succ_priority = 0.0
        for s in successors[i]:
            p = get_priority(s)
            if p > max_succ_priority:
                max_succ_priority = p
        
        res = times[i] + max_succ_priority
        memo[i] = res
        return res

    priorities = np.zeros(n, dtype=float)
    for i in range(n):
        priorities[i] = get_priority(i)
        
    return priorities
# EVOLVE-BLOCK-END
