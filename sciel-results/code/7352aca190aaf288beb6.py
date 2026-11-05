"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor graph
    successors = defaultdict(list)
    
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            
    # Calculate Total Successor Time (TST)
    # TST[i] = times[i] + sum(TST[s] for s in successors[i])
    # This metric (Weighted Precedence) captures the total workload
    # in the subgraph downstream of task i.
    memo = {}
    
    def get_tst(i):
        if i in memo:
            return memo[i]
        
        total = times[i]
        for s in successors[i]:
            total += get_tst(s)
            
        memo[i] = total
        return total

    priorities = np.zeros(n, dtype=float)
    for i in range(n):
        priorities[i] = get_tst(i)
        
    return priorities
# EVOLVE-BLOCK-END
