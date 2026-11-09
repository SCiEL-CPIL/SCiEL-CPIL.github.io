"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_priorities(instance):
    """
    Compute a composite priority for each task in an assembly line instance.
    
    The priority combines:
    1. Longest path to end (successor rank / critical path weight)
    2. Smooth bottleneck density (soft surrogate for min-stations)
    3. Transitive successor count (breadth of downstream impact)
    
    Returns an array of scores, one per task, where higher = more urgent.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Parse predecessors
    preds = instance.get("predecessors", [])
    
    # Build successor adjacency
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    if isinstance(preds, dict):
        for i, plist in preds.items():
            for p in plist:
                succ[p].append(i)
                in_degree[i] += 1
    elif isinstance(preds, (list, tuple)):
        for i, plist in enumerate(preds):
            for p in plist:
                succ[p].append(i)
                in_degree[i] += 1
    else:
        pass
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If there's a cycle, fall back to just duration
    if len(topo_order) < n:
        return times.copy()
    
    # Reverse topological order for computing longest path to end
    longest_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if succ[u]:
            longest_to_end[u] = times[u] + np.max([longest_to_end[v] for v in succ[u]])
        else:
            longest_to_end[u] = times[u]
    
    # Count transitive successors (all reachable downstream nodes)
    num_trans_succ = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for s in succ[node]:
                    if s not in visited:
                        stack.append(s)
        num_trans_succ[i] = len(visited)
    
    # Compute work remaining (sum of times of task and all transitive successors)
    # Use reverse topological order for O(N) DP
    work = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        work[u] = times[u]
        for v in succ[u]:
            work[u] += work[v] - times[v]  # avoid double-counting: work[v] already includes times[v]
        # Actually simpler: work[u] = times[u] + sum of (work[v] - times[v]) for v in succ[u]
        # But that's not right either. Let's just do: work[u] = times[u] + sum(work[v] for v in succ[u]) - sum(times[v] for v in succ[u])
        # Hmm, that's getting complicated. Let me just use the BFS approach since N=30.
    
    # Simpler: BFS for work (N=30, so O(N^2) is fine)
    work = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set([i])
        stack = [i]
        total_work = 0.0
        while stack:
            node = stack.pop()
            total_work += times[node]
            for s in succ[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        work[i] = total_work
    
    # Compute smooth bottleneck density
    # Use soft-min surrogate: instead of hard min_stations = ceil(work/cycle_time),
    # use a smooth approximation that penalizes tasks where work >> cycle_time
    cycle_time = instance.get("cycle_time", 40.0)
    if cycle_time <= 0:
        cycle_time = 40.0
    
    # Smooth surrogate for min stations: use log-sum-exp style soft-min
    # min_stations_smooth = -log(sum(exp(-x))) where x = work/cycle_time
    # But we want something that behaves like ceil(work/cycle_time) smoothly.
    # Alternative: use work/cycle_time directly as a smooth proxy for load pressure
    # The key insight: tasks with high work remaining relative to cycle time
    # create bottlenecks. Use a smooth saturation function.
    
    # Load ratio: how many cycle-times of work are downstream
    load_ratio = work / cycle_time
    
    # Smooth bottleneck: task_time * load_ratio / (1 + load_ratio)
    # This saturates as load_ratio grows, avoiding the discontinuity of ceil()
    # and giving higher priority to tasks that are both long AND have heavy downstream load
    bottleneck = times * load_ratio / (1.0 + load_ratio)
    
    # Normalize components to [0, 1] range for combination
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lte = normalize(longest_to_end)
    norm_bottleneck = normalize(bottleneck)
    norm_trans_succ = normalize(num_trans_succ)
    
    # Weighted combination
    # Heavily weight longest path to end, then smooth bottleneck density, then transitive breadth
    priority_scores = 1.0 * norm_lte + 0.5 * norm_bottleneck + 0.1 * norm_trans_succ
    
    return priority_scores

def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority
    eligible task that fits the cycle time.
    
    Returns one score per task (higher = more urgent).
    """
    start_time = time.time()
    
    try:
        scores = _compute_priorities(instance)
    except Exception:
        # Fallback to duration-based priority
        times = np.asarray(instance["times"], dtype=float)
        scores = times.copy()
    
    # Deadline guard: if we've used too much time, return what we have
    elapsed = time.time() - start_time
    if elapsed > 10:  # 10 second safety margin
        pass  # We already have a result, just return it
    
    return scores
# EVOLVE-BLOCK-END
