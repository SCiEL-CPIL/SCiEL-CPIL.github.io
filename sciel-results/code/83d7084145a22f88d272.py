"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def priority(instance):
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Longest path from each activity to any sink
    longest_to_end = np.zeros(n)
    for _ in range(n):
        for i in range(n - 1, -1, -1):
            if successors[i]:
                longest_to_end[i] = durations[i] + max(longest_to_end[j] for j in successors[i])
            else:
                longest_to_end[i] = durations[i]
    
    # Resource intensity
    if demands.ndim == 2 and demands.shape[1] > 0:
        resource_intensity = np.max(demands / capacities[np.newaxis, :], axis=1)
    else:
        resource_intensity = np.ones(n)
    
    # Successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Normalize
    if longest_to_end.max() > 0:
        crit_norm = longest_to_end / longest_to_end.max()
    else:
        crit_norm = np.ones(n)
    
    if resource_intensity.max() > 0:
        ri_norm = resource_intensity / resource_intensity.max()
    else:
        ri_norm = np.zeros(n)
    
    if succ_count.max() > 0:
        sc_norm = succ_count / succ_count.max()
    else:
        sc_norm = np.zeros(n)
    
    # Weighted combination
    priority_scores = 0.5 * crit_norm + 0.3 * ri_norm + 0.2 * sc_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
