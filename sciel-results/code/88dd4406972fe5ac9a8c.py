import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 5.0  # safety margin well under 120s
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build adjacency structures
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
                in_degree[i] += 1
    
    # Topological order using Kahn's algorithm
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    # If topological sort didn't cover all nodes (cycle), fall back
    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)
    
    # Compute Total Downstream Work (TDW) in reverse topological order
    tdw = [0.0] * n
    for i in reversed(topo_order):
        tdw[i] = times[i] + sum(tdw[j] for j in successors[i])
    
    # Compute Longest Successor Path (LSP) in reverse topological order
    lsp = [0.0] * n
    for i in reversed(topo_order):
        max_succ_lsp = 0.0
        for j in successors[i]:
            if lsp[j] > max_succ_lsp:
                max_succ_lsp = lsp[j]
        lsp[i] = times[i] + max_succ_lsp
    
    # Compute Total Upstream Work (TUW) in forward topological order
    tuw = [0.0] * n
    for i in topo_order:
        max_pred_tuw = 0.0
        if predecessors[i]:
            for pred in predecessors[i]:
                if tuw[pred] > max_pred_tuw:
                    max_pred_tuw = tuw[pred]
        tuw[i] = max_pred_tuw + times[i]
    
    # Compute Longest Predecessor Path (LPP) in forward topological order
    lpp = [0.0] * n
    for i in topo_order:
        max_pred_lpp = 0.0
        if predecessors[i]:
            for pred in predecessors[i]:
                if lpp[pred] > max_pred_lpp:
                    max_pred_lpp = lpp[pred]
        lpp[i] = max_pred_lpp + times[i]
    
    # Number of direct successors and predecessors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_predecessors = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Transitive successor count (number of tasks that depend on this task)
    trans_succ_count = [0] * n
    for i in reversed(topo_order):
        trans_succ_count[i] = len(successors[i]) + sum(trans_succ_count[j] for j in successors[i])
    trans_succ_count = np.array(trans_succ_count, dtype=float)
    
    # Transitive predecessor count
    trans_pred_count = [0] * n
    for i in topo_order:
        trans_pred_count[i] = len(predecessors[i]) + sum(trans_pred_count[p] for p in predecessors[i])
    trans_pred_count = np.array(trans_pred_count, dtype=float)
    
    # Bottleneck slack heuristic: how much of the cycle time this task consumes
    # relative to the branching it unlocks
    # Tasks that consume a large fraction of cycle time but unlock many successors
    # should be prioritized to fill stations tightly
    capacity_ratio = times / max(cycle_time, 1e-6)
    branching_factor = np.maximum(num_successors, 1.0)
    
    # Smooth bottleneck score: high when task is large relative to cycle and unlocks work
    # Use a soft-max-like combination
    bottleneck_score = capacity_ratio * (1.0 + 0.3 * np.log1p(trans_succ_count))
    
    # Tightness measure: how constrained the task is by predecessors
    # Tasks with many predecessors are "tighter" and should be scheduled earlier
    tightness = trans_pred_count / max(n, 1)
    
    # Combine heuristics with tuned weights
    # Primary: TDW (total downstream work) - tasks supporting more work go first
    # Secondary: LSP (longest successor path) - critical path tasks go first
    # Tertiary: bottleneck score - large, high-branching tasks fill stations well
    # Quaternary: tightness - tasks with many predecessors need to be scheduled early
    # Quinary: transitive successor count - unlock more work
    
    tdw_arr = np.asarray(tdw, dtype=float)
    lsp_arr = np.asarray(lsp, dtype=float)
    
    # Normalize components for stable weighting
    tdw_max = np.max(tdw_arr) if np.max(tdw_arr) > 0 else 1.0
    lsp_max = np.max(lsp_arr) if np.max(lsp_arr) > 0 else 1.0
    bottleneck_max = np.max(bottleneck_score) if np.max(bottleneck_score) > 0 else 1.0
    tightness_max = np.max(tightness) if np.max(tightness) > 0 else 1.0
    trans_succ_max = np.max(trans_succ_count) if np.max(trans_succ_count) > 0 else 1.0
    
    priorities = (
        1.0 * (tdw_arr / tdw_max) +
        0.4 * (lsp_arr / lsp_max) +
        0.3 * (bottleneck_score / bottleneck_max) +
        0.15 * (tightness / tightness_max) +
        0.1 * (trans_succ_count / trans_succ_max) +
        times * 1e-6
    )
    
    return priorities
