"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_predecessor_counts(times, predecessors):
    """Compute for each task the number of transitive predecessors (upstream depth)."""
    n = len(times)
    # Build adjacency: pred -> successors
    succ = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i]
        pred_count[i] = len(preds)
        for p in preds:
            succ[p].append(i)
    
    # Topological order via Kahn's algorithm
    in_degree = pred_count[:]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Longest path from sources (earliest start time in time units)
    # earliest[i] = max over preds of (earliest[p] + times[p]), or 0 if no preds
    earliest = [0.0] * n
    for u in topo:
        for v in succ[u]:
            val = earliest[u] + times[u]
            if val > earliest[v]:
                earliest[v] = val
    
    # Also compute transitive successor count (number of tasks that transitively depend on i)
    # Use reverse topological order
    trans_succ_count = [0] * n
    for u in reversed(topo):
        total = len(succ[u])
        for v in succ[u]:
            total += trans_succ_count[v]
        trans_succ_count[u] = total
    
    return earliest, trans_succ_count, topo, succ


def priority(instance):
    """
    Score each assembly task for the priority rule.
    Strategy: combine positional criticality (earliest start / cycle time) with
    transitive successor count (bottleneck) and task duration.
    
    The idea: tasks that are early in the timeline (low earliest start) AND have many
    transitive successors are the most critical for enabling parallelism.
    We use a composite score that rewards:
      1. High transitive successor count (more downstream work depends on it)
      2. Longer task duration (harder to pack)
      3. Low earliest start time (constrains the start phase)
    
    We normalize each component and combine them.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    t0 = time.time()
    deadline = t0 + 12.0  # safety margin well under 120s
    
    # Compute earliest start times and transitive successor counts
    earliest, trans_succ_count, topo, succ = _compute_predecessor_counts(times, predecessors)
    earliest = np.asarray(earliest, dtype=float)
    trans_succ_count = np.asarray(trans_succ_count, dtype=float)
    
    # Normalize components to [0, 1]
    # Earliest start: lower is more critical (start phase), so invert
    e_min, e_max = earliest.min(), earliest.max()
    if e_max > e_min:
        pos_crit = (e_max - earliest) / (e_max - e_min)  # 1 = earliest possible, 0 = latest
    else:
        pos_crit = np.ones(n)
    
    # Transitive successor count: higher is more critical
    ts_min, ts_max = trans_succ_count.min(), trans_succ_count.max()
    if ts_max > ts_min:
        succ_norm = (trans_succ_count - ts_min) / (ts_max - ts_min)
    else:
        succ_norm = np.ones(n)
    
    # Task duration: longer tasks are harder to place
    t_min, t_max = times.min(), times.max()
    if t_max > t_min:
        dur_norm = (times - t_min) / (t_max - t_min)
    else:
        dur_norm = np.ones(n)
    
    # Combine: weight successor count heavily (it was the best measured improvement),
    # then duration, then positional criticality
    # Use a weighted sum with emphasis on successors and duration
    score = 0.5 * succ_norm + 0.3 * dur_norm + 0.2 * pos_crit
    
    # Check deadline
    if time.time() > deadline:
        # Fallback to simple duration-based if we're running out of time
        return times.astype(float)
    
    return score
# EVOLVE-BLOCK-END
