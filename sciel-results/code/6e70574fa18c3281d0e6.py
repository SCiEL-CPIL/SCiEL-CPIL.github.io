"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    
    Strategy: "bottleneck slack" heuristic using transitive successors for
    downstream criticality, combined with processing time and depth.
    Uses a smooth soft-max surrogate for depth computation.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 100.0  # safety margin before 120s limit
    
    # Build immediate successor lists
    imm_succ = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        if preds is not None:
            for p in preds:
                imm_succ[p].append(i)
    
    # Compute transitive successor counts using BFS/DFS from each node
    # For 30 tasks, this is fast
    trans_succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = list(imm_succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(imm_succ[node])
        trans_succ_count[i] = len(visited)
    
    # Compute depth with soft-max surrogate for smoothness
    # depth[i] = 1 + soft_max(depth[p] for p in preds)
    # soft_max(x) = (1/beta) * log(sum(exp(beta * x)))
    # For small n, we can use exact max but smooth it slightly
    depth = np.zeros(n, dtype=float)
    beta = 2.0  # softness parameter
    
    for i in range(n):
        preds = predecessors[i]
        if preds is not None and len(preds) > 0:
            pred_depths = np.array([depth[p] for p in preds], dtype=float)
            # Soft-max: smooth approximation of max
            shifted = beta * pred_depths
            max_shifted = np.max(shifted)
            log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            soft_max_val = log_sum_exp / beta
            depth[i] = 1.0 + soft_max_val
        else:
            depth[i] = 0.0
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: just use times
        return times.copy()
    
    # Compute priority score
    # Base: longer tasks first
    base = times.copy()
    
    # Slack term using transitive successors (measured improvement)
    # (cycle_time - time_i) / (trans_succ_count_i + 1)
    # Smaller slack ratio = more "tight" = higher priority
    slack = (cycle_time - times) / (trans_succ_count + 1.0)
    
    # Depth bonus: deeper tasks have less flexibility
    depth_bonus = depth * 0.6
    
    # Combine: high base - low slack + high depth_bonus = high priority
    # Tune weights: emphasize processing time and depth more
    priority_scores = 1.2 * base - 0.4 * slack + depth_bonus
    
    return priority_scores
# EVOLVE-BLOCK-END
