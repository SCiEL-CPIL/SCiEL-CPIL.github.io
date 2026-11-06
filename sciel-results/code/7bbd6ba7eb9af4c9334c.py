"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    
    Strategy: Combined heuristic using:
    - Downstream subtree weight (total time of all successors + self)
    - Bottleneck slack ratio (capacity consumed vs. branching unlocked)
    - Smooth max-based depth for tie-breaking
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists and counts
    succ_list = [[] for _ in range(n)]
    succ_count = np.zeros(n, dtype=float)
    depth = np.zeros(n, dtype=float)
    
    for i in range(n):
        preds = predecessors[i]
        if preds is not None and len(preds) > 0:
            for p in preds:
                succ_list[p].append(i)
                succ_count[p] += 1
            # Smooth max using log-sum-exp for depth computation
            pred_depths = np.array([depth[p] for p in preds], dtype=float)
            # Soft max: log(sum(exp(pred_depths))) with temperature
            max_d = np.max(pred_depths)
            if max_d > 0:
                soft_max = max_d + np.log(np.sum(np.exp(pred_depths - max_d)))
            else:
                soft_max = 0.0
            depth[i] = 1.0 + soft_max
        else:
            depth[i] = 0.0
    
    # Compute downstream subtree weight (total time of task + all successors)
    # Process in reverse topological order
    subtree_weight = times.copy()
    for i in range(n - 1, -1, -1):
        for s in succ_list[i]:
            subtree_weight[i] += subtree_weight[s]
    
    # Normalize subtree weight to be comparable to times
    max_subtree = np.max(subtree_weight)
    if max_subtree > 0:
        norm_subtree = subtree_weight / max_subtree
    
    # Slack term: (cycle_time - time_i) / (succ_count_i + 1)
    # Smaller slack ratio means the task is more "tight"
    slack = (cycle_time - times) / (succ_count + 1.0)
    
    # Depth bonus: deeper tasks have less flexibility
    depth_bonus = depth * 0.3
    
    # Combine: 
    # - High downstream weight (global impact)
    # - Low slack (tight capacity)
    # - High depth (less flexibility)
    priority_scores = (
        1.5 * norm_subtree + 
        0.8 * (times / (np.max(times) + 1e-9)) - 
        0.4 * (slack / (cycle_time + 1e-9)) + 
        depth_bonus
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
