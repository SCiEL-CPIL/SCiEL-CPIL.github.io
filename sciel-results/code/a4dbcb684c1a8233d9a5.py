"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import defaultdict, deque


def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    Higher score = higher priority (picked first among eligible tasks).
    
    Strategy: Combine downstream subtree weight, bottleneck-slack ratio,
    and a light greedy-simulation tightness boost.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Build successor and predecessor maps
    # predecessors[i] = list of tasks that must come before i
    pred = defaultdict(list)
    succ = defaultdict(list)
    
    if "predecessors" in instance:
        preds_raw = instance["predecessors"]
        # preds_raw[i] is the list of predecessors of task i
        for i in range(n):
            for p in preds_raw[i]:
                pred[i].append(p)
                succ[p].append(i)
    
    # Compute downstream subtree weight (sum of times in the subtree rooted at each node)
    # Use topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(pred[i])
    
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Fallback: just use times
        return times.copy()
    
    # Downstream weight: sum of times of all successors (transitively)
    # Process in reverse topological order
    downstream_weight = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        downstream_weight[u] = times[u] + sum(downstream_weight[v] for v in succ[u])
    
    # Number of immediate successors
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Bottleneck slack ratio: (cycle_time - time_i) / (num_succ + 1)
    # Smaller ratio = tighter fit relative to branching = higher priority
    # We want to favor tasks that consume most capacity relative to branching
    # So priority should be inversely related to this ratio, or we use:
    # slack_ratio = (cycle_time - time_i) / (num_succ + 1)
    # Higher priority when slack_ratio is small
    slack_ratio = (cycle_time - times) / (num_succ + 1.0)
    
    # Light greedy-simulation tightness boost:
    # Simulate a quick greedy packing to identify tasks that tend to be
    # "left behind" (have few successors remaining when placed late)
    # This is a cheap proxy: tasks with high downstream weight relative to
    # their position in the precedence chain
    
    # Rank-based tightness: tasks that are "critical path" members get a boost
    # Compute longest path from start to each node and from each node to end
    longest_to_node = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succ[u]:
            longest_to_node[v] = max(longest_to_node[v], longest_to_node[u] + times[u])
    
    longest_from_node = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        longest_from_node[u] = times[u] + max((longest_from_node[v] for v in succ[u]), default=0.0)
    
    # Criticality: how much of the critical path goes through this task
    # Normalize
    total_cp = max(longest_to_node + longest_from_node) if n > 0 else 1.0
    if total_cp <= 0:
        total_cp = 1.0
    criticality = (longest_to_node + longest_from_node - times) / total_cp
    
    # Combine scores:
    # 1. Downstream weight (normalized) - favor tasks that unlock more work
    # 2. Inverse slack ratio - favor tasks that fit tightly
    # 3. Criticality boost - favor tasks on critical paths
    
    # Normalize each component to [0, 1] range for stable combination
    dw_min, dw_max = downstream_weight.min(), downstream_weight.max()
    dw_range = dw_max - dw_min
    if dw_range > 0:
        dw_norm = (downstream_weight - dw_min) / dw_range
    else:
        dw_norm = np.ones(n)
    
    sr_min, sr_max = slack_ratio.min(), slack_ratio.max()
    sr_range = sr_max - sr_min
    if sr_range > 0:
        # Lower slack ratio = higher priority, so invert
        sr_norm = 1.0 - (slack_ratio - sr_min) / sr_range
    else:
        sr_norm = np.ones(n)
    
    ct_min, ct_max = criticality.min(), criticality.max()
    ct_range = ct_max - ct_min
    if ct_range > 0:
        ct_norm = (criticality - ct_min) / ct_range
    else:
        ct_norm = np.ones(n)
    
    # Weighted combination
    # Emphasize downstream weight and slack ratio, with moderate criticality boost
    score = 0.45 * dw_norm + 0.35 * sr_norm + 0.20 * ct_norm
    
    # Add a small time-based component to break ties in favor of longer tasks
    t_min, t_max = times.min(), times.max()
    t_range = t_max - t_min
    if t_range > 0:
        t_norm = (times - t_min) / t_range
    else:
        t_norm = np.ones(n)
    
    score = 0.9 * score + 0.1 * t_norm
    
    return score
# EVOLVE-BLOCK-END
