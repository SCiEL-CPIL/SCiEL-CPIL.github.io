"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_DEADLINE_START = None
_DEADLINE_LIMIT = 115.0  # safety margin before 120s wall-clock


def _check_deadline():
    """Return True if we should stop and return best-so-far."""
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return (time.time() - _DEADLINE_START) > _DEADLINE_LIMIT


def _count_transitive(successors, n):
    """Count transitive successors for each node via reverse topological DP."""
    # We need topological order; caller provides it
    return None  # placeholder, actual computation done in priority


def priority(instance):
    """
    Score each assembly task for the greedy list scheduler.
    Stations are filled with the highest-priority eligible task that fits the cycle time.
    We want to minimize the mean number of stations over 5 instances.
    
    Strategy: Combine multiple heuristics into a single priority score.
    1. Longest processing time (LPT) - favors long tasks to pack them tightly.
    2. Longest path to end (critical path) - favors tasks that unlock long chains.
    3. Total downstream work (sum of times in reachable successors) - favors tasks 
       that unlock substantial total work.
    4. Transitive successor count - favors tasks that unlock many tasks (branching depth).
    5. Transitive predecessor count - favors tasks with deep upstream dependencies 
       (they are "bottlenecks" in the sense that delaying them delays many downstream tasks).
    6. Bottleneck slack - favors tasks that consume a lot of capacity relative to 
       branching they unlock.
    
    We combine these into a single score. The greedy scheduler picks the highest score
    among eligible tasks that fit.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # 1. Longest processing time
    lpt = times.copy()
    
    # Topological sort
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        return lpt
    
    # 2. Longest path from each task to the end (critical path)
    longest_path = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, times[v] + longest_path[v])
        longest_path[u] = max_succ
    
    # 3. Total downstream work: sum of times of all transitive successors
    total_downstream = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        s = 0.0
        for v in successors[u]:
            s += times[v] + total_downstream[v]
        total_downstream[u] = s
    
    # 4. Transitive successor count
    trans_succ_count = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        s = 0.0
        for v in successors[u]:
            s += 1.0 + trans_succ_count[v]
        trans_succ_count[u] = s
    
    # 5. Transitive predecessor count (forward pass)
    trans_pred_count = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in successors[u]:
            trans_pred_count[v] = max(trans_pred_count[v], trans_pred_count[u] + 1.0)
    
    # 6. Number of immediate successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # 7. Bottleneck slack
    eps = 1e-9
    slack = (cycle_time - times) / (num_successors + 1.0)
    bottleneck_score = 1.0 / (slack + eps)
    
    # Normalize each component to [0, 1] range
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    lpt_norm = normalize(lpt)
    lp_norm = normalize(longest_path)
    td_norm = normalize(total_downstream)
    ts_norm = normalize(trans_succ_count)
    tp_norm = normalize(trans_pred_count)
    succ_norm = normalize(num_successors)
    bottleneck_norm = normalize(bottleneck_score)
    
    # Combine with tuned weights
    # LPT: strong baseline for packing
    # LP: critical path awareness
    # TD: total downstream work (broader than LP)
    # TS: transitive successor count (depth of branching)
    # TP: transitive predecessor count (how "deep" the task is in the chain)
    # BN: bottleneck slack
    w_lpt = 0.25
    w_lp = 0.20
    w_td = 0.15
    w_ts = 0.10
    w_tp = 0.10
    w_bottleneck = 0.20
    
    priority_scores = (w_lpt * lpt_norm + 
                       w_lp * lp_norm + 
                       w_td * td_norm + 
                       w_ts * ts_norm + 
                       w_tp * tp_norm + 
                       w_bottleneck * bottleneck_norm)
    
    # Add a small tie-breaker based on original time to ensure deterministic ordering
    priority_scores = priority_scores * 1.0 + lpt * 1e-6
    
    return priority_scores
# EVOLVE-BLOCK-END
