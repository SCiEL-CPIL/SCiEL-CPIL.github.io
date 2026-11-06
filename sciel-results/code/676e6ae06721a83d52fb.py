"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n_tasks = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n_tasks)]
    in_degree = np.zeros(n_tasks, dtype=int)
    
    for i in range(n_tasks):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Calculate Weighted Successors (Sum of direct successor times + own time)
    # This heuristic prioritizes tasks that have heavy immediate successors.
    # P[i] = time[i] + sum(time[j] for j in successors[i])
    
    priorities = np.zeros(n_tasks, dtype=float)
    
    # We can compute this directly without topological sort if we only use direct successors.
    # However, to be more robust, let's compute the sum of *all* downstream times?
    # No, direct is faster and often sufficient. Let's try direct first.
    # If direct is too weak, we might need the recursive sum.
    # Let's implement the recursive sum of downstream times (Total Downstream Time - TDT).
    # TDT[i] = time[i] + sum(TDT[j] for j in successors[i]) is WRONG because of shared successors (diamonds).
    # If we want sum of unique downstream nodes, it's harder.
    # Let's stick to **Sum of Direct Successor Times** (SDST) for simplicity and speed.
    # It is distinct from BL (which uses max).
    
    for i in range(n_tasks):
        sum_succ_times = 0.0
        for j in successors[i]:
            sum_succ_times += times[j]
        priorities[i] = times[i] + sum_succ_times
        
    return priorities
# EVOLVE-BLOCK-END
