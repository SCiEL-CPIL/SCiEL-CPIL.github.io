"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    """Score each assembly task. Returns one score per task."""
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Compute in-degrees and topological order
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
    
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    # Compute longest path from start to each task (upstream depth)
    longest_up = [0.0] * n
    for u in topo:
        for v in successors[u]:
            candidate = longest_up[u] + times[u]
            if candidate > longest_up[v]:
                longest_up[v] = candidate
    
    # Compute longest path from each task to end (downstream depth)
    longest_down = [0.0] * n
    for u in reversed(topo):
        for v in successors[u]:
            candidate = times[v] + longest_down[v]
            if candidate > longest_down[u]:
                longest_down[u] = candidate
    
    # Total downstream work (sum of times of all descendants)
    total_down = [0.0] * n
    for u in reversed(topo):
        total = 0.0
        for v in successors[u]:
            total += times[v] + total_down[v]
        total_down[u] = total
    
    # Number of immediate successors
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of immediate predecessors
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Total work in direct predecessors
    pred_work = np.zeros(n, dtype=float)
    for i in range(n):
        for p in predecessors[i]:
            pred_work[i] += times[p]
    
    # Total work in all predecessors (transitive)
    total_pred_work = np.zeros(n, dtype=float)
    for u in topo:
        for v in successors[u]:
            total_pred_work[v] = max(total_pred_work[v], total_pred_work[u] + times[u])
    
    # Critical path total length
    cp_length = max(longest_up[i] + times[i] + longest_down[i] for i in range(n)) if n > 0 else 1.0
    if cp_length < 1e-9:
        cp_length = 1.0
    
    # Normalize components
    max_time = max(times) + 1e-9
    max_longest_up = max(longest_up) + 1e-9
    max_longest_down = max(longest_down) + 1e-9
    max_total_down = max(total_down) + 1e-9
    max_num_succ = max(num_succ) + 1e-9
    max_num_pred = max(num_pred) + 1e-9
    max_pred_work = max(pred_work) + 1e-9
    max_total_pred_work = max(total_pred_work) + 1e-9
    
    # Component scores
    time_score = times / max_time
    up_score = np.array(longest_up, dtype=float) / max_longest_up
    down_score = np.array(longest_down, dtype=float) / max_longest_down
    total_down_score = np.array(total_down, dtype=float) / max_total_down
    succ_score = num_succ / max_num_succ
    pred_score = num_pred / max_num_pred
    pred_work_score = pred_work / max_pred_work
    total_pred_work_score = total_pred_work / max_total_pred_work
    
    # Critical path proximity: how close is this task to being on the critical path?
    # A task is on/near CP if longest_up[t] + times[t] + longest_down[t] is close to cp_length
    cp_proximity = (np.array(longest_up, dtype=float) + times + np.array(longest_down, dtype=float)) / cp_length
    
    # Additional features
    # Positional depth: how deep is this task from the start (longest path to it)
    # This helps prioritize tasks that come later in the chain
    pos_depth = np.array(longest_up, dtype=float)
    max_pos_depth = max(pos_depth) + 1e-9
    pos_depth_score = pos_depth / max_pos_depth
    
    # Bottleneck score: tasks whose successors are also on the critical path
    # A task is a bottleneck if removing it would increase the makespan
    bottleneck = np.zeros(n, dtype=float)
    for i in range(n):
        if not successors[i]:
            bottleneck[i] = 1.0  # terminal tasks are bottlenecks
        else:
            # How many successors are on or near the critical path?
            near_cp = 0
            for v in successors[i]:
                if longest_up[v] + times[v] + longest_down[v] >= cp_length * 0.9:
                    near_cp += 1
            bottleneck[i] = near_cp / max(len(successors[i]), 1)
    max_bottleneck = max(bottleneck) + 1e-9
    bottleneck_score = bottleneck / max_bottleneck
    
    # Work-to-successor ratio: total work that depends on this task
    # Higher means this task is more "upstream" and should be placed earlier
    work_dependent = np.zeros(n, dtype=float)
    for u in reversed(topo):
        work_dependent[u] = total_down[u] + times[u]
    max_work_dependent = max(work_dependent) + 1e-9
    work_dep_score = work_dependent / max_work_dependent
    
    # Weighted combination - refined for assembly line balancing
    # Key heuristics: largest processing time, most followers, critical path proximity
    # Also: deeper tasks (more upstream work before them) and bottleneck tasks
    composite = (
        0.28 * time_score +
        0.18 * down_score +
        0.14 * cp_proximity +
        0.13 * succ_score +
        0.08 * total_down_score +
        0.06 * pos_depth_score +
        0.05 * bottleneck_score +
        0.05 * work_dep_score +
        0.03 * pred_score
    )
    
    return composite
# EVOLVE-BLOCK-END
