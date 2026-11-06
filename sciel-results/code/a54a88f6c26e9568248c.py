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
    # Use a more robust calculation that accumulates properly
    total_down = [0.0] * n
    for u in reversed(topo):
        total = 0.0
        for v in successors[u]:
            total += times[v] + total_down[v]
        total_down[u] = total
    
    # Number of transitive successors (not just immediate)
    transitive_succ_count = [0] * n
    for u in reversed(topo):
        count = 0
        for v in successors[u]:
            count += 1 + transitive_succ_count[v]
        transitive_succ_count[u] = count
    transitive_succ_arr = np.array(transitive_succ_count, dtype=float)
    max_transitive_succ = max(transitive_succ_arr) + 1e-9 if n > 0 else 1.0
    transitive_succ_score = transitive_succ_arr / max_transitive_succ
    
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
    
    # Transitive successor count score already computed above
    # Re-normalize to ensure consistency
    max_transitive_succ = max(transitive_succ_arr) + 1e-9 if n > 0 else 1.0
    transitive_succ_score = transitive_succ_arr / max_transitive_succ
    
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
    
    # Weighted combination - tuned for assembly line balancing
    # Emphasis on: time, downstream depth, transitive successors, critical path proximity
    # Findings show prioritizing tasks with more transitive successors (bottleneck tasks)
    # reduces station count by making more tasks eligible earlier for better packing
    composite = (
        0.22 * time_score +
        0.15 * down_score +
        0.10 * cp_proximity +
        0.10 * succ_score +
        0.25 * total_down_score +
        0.15 * transitive_succ_score +
        0.03 * pred_score
    )
    
    return composite
# EVOLVE-BLOCK-END
