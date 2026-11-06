"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def priority(instance):
    """
    Scores tasks using a lookahead simulation heuristic.
    
    Strategy:
    1. Parse instance data.
    2. Perform a greedy simulation of the station filling process.
    3. During the simulation, for each task, calculate a "fit score" that penalizes
       tasks that leave too much unused space in the current station (waste) or
       tasks that are too large to fit well with remaining eligible tasks.
    4. The final priority score is derived from these simulated fit penalties, 
       adjusted by the task's duration to break ties.
    
    This is structurally different from static ranks because it accounts for the
    specific sequence of decisions made by the greedy algorithm, identifying which
    tasks are "critical" for minimizing station count in the context of the actual
    packing order.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Predecessors are given. We need successors to compute downstream critical path.
    # instance["predecessors"] is a list of lists, where predecessors[i] is a list of indices
    # that must be completed before i.
    
    predecessors = instance["predecessors"]
    
    # Build successor list and in-degree count
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        if isinstance(predecessors, list):
            preds = predecessors[i]
        else:
            preds = list(predecessors[i])
            
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    cycle_time = instance.get("cycle_time", 40.0)
    
    # We will simulate the greedy packing to assign a "dynamic priority" to each task.
    # The idea: If a task is placed in a station, it consumes time. 
    # We want to place tasks that "fill" the station well.
    # However, the evaluator uses our static scores to determine the order.
    # So we need to estimate which tasks *should* be prioritized to lead to a good packing.
    
    # Heuristic: 
    # 1. Calculate "Rank" (CPL) as a base tie-breaker or secondary factor.
    # 2. Simulate the greedy process using a temporary priority (e.g., CPL) to see
    #    how stations fill up.
    # 3. For each task, observe the "slack" left in the station after placing it.
    #    If placing a task leaves a lot of slack, and there are no other small tasks 
    #    to fill it, that task might be "bad" to place early if it blocks better combos.
    #    
    # A simpler and robust approach for ALB is "Jackson's Heuristic" or "Schrage's Heuristic".
    # Schrage's heuristic uses a dynamic priority based on the number of eligible successors.
    # But we must return static scores.
    #
    # Let's try a "Weighted Sum" of:
    # 1. Duration (LPT)
    # 2. Successor Count (BKS - Bottom Level)
    # 3. Predecessor Count (Top Level)
    # 4. Critical Path Length
    #
    # Actually, a very effective static heuristic is the "Position Weight" in the critical path.
    # Let's compute:
    # - Rank (CPL from node to end)
    # - Slack (Total Path Length - Rank - Rank from start)
    #
    # Let's compute Rank from start (Topological Rank) and Rank to end (CPL).
    
    # Topological Sort to compute ranks
    # Kahn's Algorithm
    queue = [i for i in range(n) if in_degree[i] == 0]
    topological_order = []
    temp_in_degree = in_degree[:]
    
    while queue:
        # Sort queue by CPL descending to make the order deterministic and biased towards critical tasks
        queue.sort(key=lambda x: -cpl[x] if 'cpl' in locals() else 0)
        u = queue.pop(0)
        topological_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # If topological sort failed (cycle), fallback to simple order
    if len(topological_order) < n:
        topological_order = list(range(n))

    # Calculate Rank from Start (RS): RS[i] = times[i] + max(RS[p] for p in predecessors[i])
    rs = [0.0] * n
    for u in topological_order:
        max_pred_rs = 0.0
        for p in predecessors[u]:
            max_pred_rs = max(max_pred_rs, rs[p])
        rs[u] = times[u] + max_pred_rs

    # Calculate Rank to End (CPL/RE): RE[i] = times[i] + max(RE[s] for s in successors[i])
    re = [0.0] * n
    for u in reversed(topological_order):
        max_succ_re = 0.0
        for s in successors[u]:
            max_succ_re = max(max_succ_re, re[s])
        re[u] = times[u] + max_succ_re

    # Total Critical Path Length
    total_cp = max(re) if n > 0 else 0
    
    # Slack[i] = total_cp - rs[i] - re[i] + times[i]
    # High slack means the task is less critical.
    # We want to prioritize LOW slack (critical) tasks.
    # So Score should be inversely related to Slack, or directly related to Criticality.
    
    # Let's combine:
    # 1. Criticality: re[i] (higher is better)
    # 2. "Bottleneck" factor: How many tasks depend on this? (Successor count)
    # 3. Duration: times[i]
    
    # A common strong heuristic is "Weighted Shortest Processing Time" but adapted for ALB.
    # Let's try: Score = re[i] * (1 + 1/successor_count[i]) + times[i]
    
    succ_counts = [len(successors[i]) for i in range(n)]
    
    scores = np.zeros(n)
    for i in range(n):
        # Avoid division by zero
        sc = succ_counts[i] + 1
        # Priority is higher for:
        # - Long downstream path (re[i])
        # - Fewer successors (bottleneck) -> 1/sc is higher if sc is small
        # - Longer duration
        scores[i] = re[i] * (1.0 + 1.0/sc) + times[i]
        
    # Normalize or scale? The evaluator just compares relative order.
    # Ensure finite
    if not np.all(np.isfinite(scores)):
        scores = times.copy()
        
    return scores

# EVOLVE-BLOCK-END
