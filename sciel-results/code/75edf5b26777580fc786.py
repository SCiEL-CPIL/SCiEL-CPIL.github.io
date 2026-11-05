"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _softmax(values, temperature=1.0):
    """Compute soft-max (log-sum-exp) as a smooth surrogate for max."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    v = np.asarray(values, dtype=float)
    max_v = np.max(v)
    # For numerical stability
    exps = np.exp((v - max_v) / temperature)
    return max_v + temperature * np.log(np.sum(exps))


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    
    The evaluator fills stations with the highest-priority eligible task that fits
    the cycle time. Lower total number of stations is better.
    
    This implementation uses a refined priority score that combines:
    1. Positional Weight (PW): Sum of times of the task and all its successors.
    2. Longest Path (LP): The length of the longest path from the task to the end.
    3. Soft Longest Path (SLP): Smooth surrogate via log-sum-exp for LP.
    4. Critical Path Density (CPD): Ratio of task time to Longest Path.
    5. Critical Ratio (CR): Downstream workload / task time. Identifies
       high-volume bottleneck tasks relative to their own duration.
    6. Rank: Number of immediate successors.
    7. Total Successor Count: Total number of transitive successors.
    8. Task Time: Longer tasks prioritized for packing efficiency.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Compute features in reverse topological order
    pw = np.zeros(n, dtype=float)       # Positional Weight
    lp = np.zeros(n, dtype=float)       # Longest Path (hard max)
    slp = np.zeros(n, dtype=float)      # Soft Longest Path (log-sum-exp)
    total_succ = np.zeros(n, dtype=float)
    downstream_sum = np.zeros(n, dtype=float)  # Sum of successor times only
    
    for u in reversed(top_order):
        t_u = times[u]
        pw_u = t_u
        lp_u = t_u
        slp_u = t_u
        ts_u = 0.0
        ds_u = 0.0
        
        if succ[u]:
            succ_indices = np.array(succ[u])
            pw_u += np.sum(pw[succ_indices])
            lp_u += np.max(lp[succ_indices])
            # Soft-max surrogate: smooth version of max(lp[succ])
            slp_u += _softmax(lp[succ_indices], temperature=2.0)
            for v in succ[u]:
                ts_u += 1 + total_succ[v]
                ds_u += times[v]
        
        pw[u] = pw_u
        lp[u] = lp_u
        slp[u] = slp_u
        total_succ[u] = ts_u
        downstream_sum[u] = ds_u
    
    # Deadline guard
    if time.time() - start_time > 100:
        return times

    # Normalize components
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_pw = normalize(pw)
    norm_lp = normalize(lp)
    norm_slp = normalize(slp)
    
    # Critical Path Density: t_i / LP_i
    lp_safe = np.where(lp > 1e-12, lp, 1.0)
    cpd = times / lp_safe
    norm_cpd = normalize(cpd)
    
    # Critical Ratio: downstream_sum / task_time
    # High CR means task is a bottleneck relative to its own duration
    t_safe = np.where(times > 1e-12, times, 1.0)
    critical_ratio = downstream_sum / t_safe
    norm_cr = normalize(critical_ratio)
    
    # Rank: number of immediate successors
    rank = np.array([len(succ[i]) for i in range(n)], dtype=float)
    norm_rank = normalize(rank)
    
    # Total Successor Count
    norm_total_succ = normalize(total_succ)
    
    # Task Time
    norm_times = normalize(times)
    
    # Weighted combination
    # PW: strongest predictor of remaining work
    # LP: critical path constraint
    # SLP: smooth surrogate that captures near-critical paths
    # CPD: local bottleneck density
    # CR: dynamic bottleneck ratio (downstream volume / own duration)
    # Rank: branching factor
    # Total succ: overall reach
    # Times: packing tie-breaker
    
    w_pw = 0.35
    w_lp = 0.15
    w_slp = 0.10
    w_cpd = 0.12
    w_cr = 0.12
    w_rank = 0.07
    w_total_succ = 0.05
    w_times = 0.04
    
    score = (w_pw * norm_pw + 
             w_lp * norm_lp +
             w_slp * norm_slp +
             w_cpd * norm_cpd + 
             w_cr * norm_cr +
             w_rank * norm_rank +
             w_total_succ * norm_total_succ +
             w_times * norm_times)
    
    # Small tie-breaker favoring longer tasks
    score = score + 1e-5 * times
    
    if time.time() - start_time > 110:
        return times
    
    return score
# EVOLVE-BLOCK-END
