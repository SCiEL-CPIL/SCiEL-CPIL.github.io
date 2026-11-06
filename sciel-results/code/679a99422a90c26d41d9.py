"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import time


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", [set() for _ in range(n)])
    cycle_time = instance.get("cycle_time", 40)
    
    # Compute number of successors (reverse topological order)
    succs = [set() for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].add(i)
    
    # Count successors using DFS with memoization
    @lru_cache(maxsize=None)
    def count_succs(i):
        if not succs[i]:
            return 0
        return 1 + sum(count_succs(j) for j in succs[i])
    
    # Compute number of predecessors
    pred_counts = [len(preds[i]) for i in range(n)]
    
    # Compute total time of successors (including self)
    @lru_cache(maxsize=None)
    def total_time_succ(i):
        return times[i] + sum(total_time_succ(j) for j in succs[i])
    
    # Priority: combine longest processing time, number of successors, 
    # and total downstream time to break ties and guide station filling
    # Higher priority = processed first
    succ_counts = np.array([count_succs(i) for i in range(n)])
    total_times = np.array([total_time_succ(i) for i in range(n)])
    
    # Normalize components
    max_time = times.max() if times.max() > 0 else 1
    max_succ = succ_counts.max() if succ_counts.max() > 0 else 1
    max_total = total_times.max() if total_times.max() > 0 else 1
    
    # Weighted combination: prioritize long tasks, then many successors, then total downstream time
    priority_scores = (
        0.4 * (times / max_time) +
        0.3 * (succ_counts / max_succ) +
        0.3 * (total_times / max_total)
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
