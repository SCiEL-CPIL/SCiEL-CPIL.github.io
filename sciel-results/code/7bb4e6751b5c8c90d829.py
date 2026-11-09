"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    The heuristic picks the highest-priority eligible task that fits the cycle time.
    We want to minimize the number of stations, so we want to pack tasks tightly.
    
    Strategy:
    1. Compute a base priority based on task duration (longer tasks first) and
       the number of successors (tasks with many successors should be placed early
       to allow their dependents to be placed in later stations).
    2. Use a combination of:
       - Task duration (higher is better for packing)
       - Number of immediate successors (higher means more urgent to place)
       - Depth in the DAG (tasks closer to the end should be prioritized)
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor counts
    succ_counts = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            succ_counts[p] += 1
    
    # Compute depth (longest path from start to each task)
    # depth[i] = length of longest path ending at task i
    depth = np.zeros(n, dtype=float)
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    # Kahn's algorithm for topological sort
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    in_deg = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in range(n):
            if u in predecessors[v]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
    
    # Compute depth in topological order
    for u in topo_order:
        for v in range(n):
            if u in predecessors[v]:
                depth[v] = max(depth[v], depth[u] + times[u])
    
    # Compute remaining path length (longest path from each task to end)
    # Use reverse topological order
    rem_path = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        for v in range(n):
            if u in predecessors[v]:
                rem_path[u] = max(rem_path[u], rem_path[v] + times[v])
    
    # Priority: combine multiple factors
    # - Higher task duration -> higher priority (pack large tasks first)
    # - Higher successor count -> higher priority (place tasks with many dependents early)
    # - Lower remaining path -> higher priority (tasks near the end can be placed later, 
    #   so prioritize tasks with less remaining work after them)
    
    # Normalize each component
    max_time = np.max(times) if n > 0 else 1
    max_succ = np.max(succ_counts) if n > 0 else 1
    max_depth = np.max(depth) if n > 0 else 1
    max_rem = np.max(rem_path) if n > 0 else 1
    
    # Weighted combination
    # We want tasks that are:
    # 1. Long (to pack efficiently)
    # 2. Have many successors (to unlock dependents early)
    # 3. Have short remaining path (they are near the end, so placing them doesn't delay much)
    
    norm_time = times / max_time if max_time > 0 else times
    norm_succ = succ_counts.astype(float) / max_succ if max_succ > 0 else succ_counts.astype(float)
    norm_rem = rem_path / max_rem if max_rem > 0 else rem_path
    
    # Priority = w1 * norm_time + w2 * norm_succ - w3 * norm_rem
    # We subtract norm_rem because tasks with long remaining paths should be placed earlier,
    # so they get lower "urgency" in terms of being placed last. Wait, let me reconsider.
    
    # Actually, for ALB, a common heuristic is:
    # - Position weight: tasks with more successors should be placed earlier
    # - Task time: longer tasks first
    # - Remaining time: tasks with less remaining work can be placed later
    
    # Let's use: priority = norm_time + 0.5 * norm_succ
    # This prioritizes long tasks and tasks with many successors.
    
    priority_scores = norm_time + 0.5 * norm_succ
    
    return priority_scores
# EVOLVE-BLOCK-END
