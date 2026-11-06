import numpy as np
from collections import defaultdict

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists to count immediate successors
    successors = defaultdict(list)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            
    # Calculate number of immediate successors for each task
    imm_succ_counts = np.zeros(n, dtype=float)
    for i in range(n):
        imm_succ_counts[i] = len(successors[i])
        
    # Calculate slack
    slack = np.maximum(cycle_time - times, 1e-6)
    
    # Calculate bottleneck slack ratio: slack / (immediate successors + 1)
    # Favors tasks with small slack and high branching (many immediate successors)
    denom = imm_succ_counts + 1.0
    ratio = slack / denom
    
    # Score: Inverse of the ratio. Higher score means higher priority.
    scores = 1.0 / (ratio + 1e-6)
    
    return scores
