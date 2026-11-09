"""Assembly Line Balancing: bottleneck-density priority using downstream critical path and station scarcity."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque


def _build_graph(instance):
    n = len(instance["times"])
    preds = instance["predecessors"]
    # successors[j] = list of tasks that have j as a predecessor
    succ = [[] for _ in range(n)]
    indeg = [0] * n
    for j in range(n):
        ps = preds[j] if preds is not None else []
        if ps is None:
            ps = []
        for p in ps:
            succ[p].append(j)
            indeg[j] += 1
    return succ, indeg


def _compute_topo_order(succ, n):
    """Return topological order of tasks."""
    indeg = [0] * n
    for j in range(n):
        for s in succ[j]:
            indeg[s] += 1
    q = deque([j for j in range(n) if indeg[j] == 0])
    topo = []
    while q:
        j = q.popleft()
        topo.append(j)
        for s in succ[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    return topo


def _longest_path_to_end(succ, times, n, topo):
    """lp[j] = times[j] + max(lp[s] for s in succ[j])"""
    lp = np.zeros(n, dtype=float)
    for j in reversed(topo):
        best = 0.0
        for s in succ[j]:
            if lp[s] > best:
                best = lp[s]
        lp[j] = times[j] + best
    # Tasks not in topo (cycle members) get just their own time
    for j in range(n):
        if lp[j] == 0.0:
            lp[j] = times[j]
    return lp


def _downstream_depth(succ, n, topo):
    """
    depth[j] = 1 + max(depth[s] for s in succ[j]), or 1 if no successors.
    This measures how many levels of tasks remain after j.
    """
    depth = np.ones(n, dtype=float)
    for j in reversed(topo):
        if succ[j]:
            max_d = max(depth[s] for s in succ[j])
            depth[j] = 1.0 + max_d
    return depth


def _descendant_count(succ, n, topo):
    """
    count[j] = number of all downstream tasks reachable from j.
    """
    count = np.zeros(n, dtype=float)
    for j in reversed(topo):
        if succ[j]:
            count[j] = 1.0 + max(count[s] for s in succ[j])
        # else count[j] stays 0
    return count


def priority(instance):
    start = time.time()
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance["predecessors"]
    cycle_time = instance.get("cycle_time", 40.0)
    
    succ, _ = _build_graph(instance)
    topo = _compute_topo_order(succ, n)
    
    # Longest path to end (critical path length)
    lp = _longest_path_to_end(succ, times, n, topo)
    
    # Downstream depth: how many levels of tasks remain
    depth = _downstream_depth(succ, n, topo)
    
    # Descendant count: total number of downstream tasks
    desc_count = _descendant_count(succ, n, topo)
    
    # Immediate successor count
    nsucc = np.array([len(s) for s in succ], dtype=float)
    
    # Bottleneck density: ratio of task time to minimum stations needed downstream
    # Min stations to complete downstream subtree ≈ ceil(lp[j] / cycle_time)
    # But we want to penalize tasks that are on tight critical paths
    # A task with high lp relative to cycle_time creates more station pressure
    
    # Normalize to [0,1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    s_lp = norm(lp)
    s_depth = norm(depth)
    s_desc = norm(desc_count)
    s_nsucc = norm(nsucc)
    s_time = norm(times)
    
    # Bottleneck density score:
    # High lp means this task's completion delays many downstream tasks
    # High depth means many sequential stages remain
    # High descendant count means many tasks depend on this one
    # 
    # Key insight: prioritize tasks that are BOTH on the critical path AND
    # have significant downstream impact. This creates a "bottleneck" signal.
    #
    # Use a multiplicative interaction: tasks that are critical AND bottlenecked
    # get boosted, while tasks that are only one or the other get moderate scores.
    
    # Primary: critical path length (most important for station count)
    # Secondary: bottleneck density (time * downstream pressure)
    # Tertiary: structural importance (depth, descendants)
    
    # Bottleneck density = time[j] * (1 + depth[j]) / cycle_time
    # This captures: longer tasks on deeper paths create more station pressure
    bottleneck_density = times * (1.0 + depth) / cycle_time
    s_bd = norm(bottleneck_density)
    
    # Combine with weighted sum emphasizing critical path and bottleneck density
    score = 0.45 * s_lp + 0.30 * s_bd + 0.10 * s_depth + 0.08 * s_desc + 0.07 * s_nsucc
    
    # Safety margin guard
    if time.time() - start > 100:
        pass
    
    return score
# EVOLVE-BLOCK-END
