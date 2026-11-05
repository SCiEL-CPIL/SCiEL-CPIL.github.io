"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    """
    Structural change: Use 'Total Downstream Work' (sum of all descendant times)
    combined with 'Inverse Slack' (how tight the task is relative to cycle time)
    and 'Topological Depth' as a tie-breaker.
    
    Total downstream work captures the total amount of work that depends on this task,
    which is different from just the longest path. A task with many parallel successors
    has high total downstream work even if its longest path is moderate.
    
    Inverse slack prioritizes tasks that have little slack relative to the cycle time,
    as these are more likely to be bottlenecks.
    """
    start_time = time.time()
    DEADLINE = 120.0

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40)
    predecessors = instance.get("predecessors", [])

    # Build adjacency
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)

    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort via Kahn's algorithm
    topo_order = []
    in_deg = in_degree.copy()
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(topo_order) < n:
        return times.copy()

    # Compute:
    # 1. Longest downstream path (critical path)
    # 2. Total downstream work (sum of all descendant times, not just longest path)
    # 3. Topological depth (position in topo order, normalized)

    longest_downstream = np.zeros(n, dtype=float)
    total_downstream_work = np.zeros(n, dtype=float)

    for node in reversed(topo_order):
        max_succ_ld = 0.0
        sum_succ_work = 0.0
        for succ in successors[node]:
            if longest_downstream[succ] > max_succ_ld:
                max_succ_ld = longest_downstream[succ]
            sum_succ_work += total_downstream_work[succ]
        longest_downstream[node] = times[node] + max_succ_ld
        total_downstream_work[node] = times[node] + sum_succ_work

    # Topological depth: earlier in topo order = lower index
    # We want tasks that come earlier in the DAG to have higher priority for tie-breaking
    topo_index = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        topo_index[node] = idx

    # Normalize
    max_ld = np.max(longest_downstream) if n > 0 else 1.0
    max_td = np.max(total_downstream_work) if n > 0 else 1.0
    max_topo = n if n > 0 else 1.0

    if max_ld == 0:
        max_ld = 1.0
    if max_td == 0:
        max_td = 1.0
    if max_topo == 0:
        max_topo = 1.0

    # Inverse slack: how tight is this task relative to cycle time?
    # Slack = cycle_time - longest_downstream (roughly)
    # Inverse slack is higher when slack is smaller
    # But we need to be careful: if longest_downstream > cycle_time, slack is negative
    # Let's use: inverse_slack = 1 / (1 + max(0, cycle_time - longest_downstream))
    # This gives higher values when the task's downstream path is close to or exceeds cycle time

    slack = cycle_time - longest_downstream
    # Clip slack to be non-negative for the denominator
    slack_clipped = np.maximum(slack, 0.0)
    inverse_slack = 1.0 / (1.0 + slack_clipped)

    # Normalize inverse_slack
    max_is = np.max(inverse_slack) if n > 0 else 1.0
    if max_is == 0:
        max_is = 1.0
    norm_is = inverse_slack / max_is

    # Normalize other components
    norm_ld = longest_downstream / max_ld
    norm_td = total_downstream_work / max_td
    # Invert topo_index so that earlier tasks get higher values
    norm_topo = 1.0 - (topo_index / max_topo)

    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return longest_downstream.copy()

    # Combine: Total downstream work is the primary signal (captures total dependent work),
    # inverse slack is secondary (captures tightness), topo depth is a small tie-breaker
    score = 1.0 * norm_td + 0.5 * norm_is + 0.1 * norm_topo

    return score
# EVOLVE-BLOCK-END
