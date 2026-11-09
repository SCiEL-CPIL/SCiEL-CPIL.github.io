"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

_DEADLINE = 120.0
_START_TIME = None


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def priority(instance):
    """
    Computes priority scores using CPM slack (total float) combined with
    resource demand intensity and tail length.
    Lower slack (more critical) and higher resource demand / longer tail
    yield higher priority scores.
    """
    _get_start_time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Extract resource data
    if "demand" in instance:
        demands_raw = instance["demand"]
    elif "demands" in instance:
        demands_raw = instance["demands"]
    else:
        demands_raw = np.zeros((n, 0))
    demands = np.asarray(demands_raw, dtype=float)
    
    if "capacity" in instance:
        capacities_raw = instance["capacity"]
    elif "capacities" in instance:
        capacities_raw = instance["capacities"]
    else:
        capacities_raw = np.array([float('inf')])
    capacities = np.asarray(capacities_raw, dtype=float)
    
    # Handle shape edge cases
    if demands.ndim == 1:
        if demands.size == n:
            demands = demands.reshape(n, 1)
        elif demands.size > 0:
            demands = demands.reshape(1, -1)
        else:
            demands = np.zeros((n, 1))
    if demands.ndim == 2 and demands.shape[0] != n:
        if demands.shape[0] == 1:
            demands = np.tile(demands, (n, 1))
        elif demands.shape[1] == n:
            demands = demands.T
        else:
            demands = np.zeros((n, demands.shape[1] if demands.shape[1] > 0 else 1))
    
    if capacities.ndim == 0:
        capacities = np.array([float(capacities)])
    
    m = max(1, len(capacities))
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
    
    # Normalize predecessors
    preds_raw = instance.get("predecessors", instance.get("predecessor_lists", [[] for _ in range(n)]))
    predecessors = []
    for i in range(n):
        p = preds_raw[i]
        if p is None:
            predecessors.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            predecessors.append([int(x) for x in p])
        else:
            predecessors.append([int(p)])
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if 0 <= i < n:
                successors[i].append(j)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            if 0 <= p < n:
                in_degree[i] += 1
    
    # Topological order via Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Handle potential cycles: add any unvisited nodes
    if len(topo_order) < n:
        in_topo = set(topo_order)
        for i in range(n):
            if i not in in_topo:
                topo_order.append(i)
    
    # Forward pass: head[i] = longest path from start to i (excluding i's duration)
    head = np.zeros(n)
    for node in topo_order:
        best = 0.0
        for p in predecessors[node]:
            if 0 <= p < n:
                path_val = head[p] + durations[p]
                if path_val > best:
                    best = path_val
        head[node] = best
    
    # Backward pass: tail[i] = longest path from i to end (excluding i's duration)
    tail = np.zeros(n)
    for node in reversed(topo_order):
        best = 0.0
        for s in successors[node]:
            path_val = durations[s] + tail[s]
            if path_val > best:
                best = path_val
        tail[node] = best
    
    # Critical path length (unconstrained project duration)
    if n > 0:
        cp_length = float(np.max(head + durations + tail))
    else:
        cp_length = 0.0
    
    # Total float (slack) for each activity
    slack = cp_length - (head + durations + tail)
    slack = np.maximum(slack, 0.0)
    
    # Normalize slack to [0, 1] range (inverted: 0 = most critical, 1 = least critical)
    if cp_length > 0:
        norm_slack = slack / cp_length
    else:
        norm_slack = np.zeros(n)
    
    # Resource demand intensity: max utilization ratio per activity
    # For each activity, compute max(demand[r] / capacity[r]) over resources
    if m > 0 and np.any(capacities > 0):
        # Avoid division by zero
        safe_caps = np.where(capacities > 0, capacities, 1.0)
        utilization = demands / safe_caps  # shape (n, m)
        # For resources with infinite capacity, utilization is 0
        inf_mask = (capacities == float('inf'))
        if np.any(inf_mask):
            utilization[:, inf_mask] = 0.0
        max_util = np.max(utilization, axis=1) if m > 0 else np.zeros(n)
    else:
        max_util = np.zeros(n)
    
    # Normalize max_util to [0, 1]
    max_util_val = float(np.max(max_util)) if n > 0 else 0.0
    if max_util_val > 0:
        norm_util = max_util / max_util_val
    else:
        norm_util = np.zeros(n)
    
    # Tail length normalized
    max_tail = float(np.max(tail)) if n > 0 else 0.0
    if max_tail > 0:
        norm_tail = tail / max_tail
    else:
        norm_tail = np.zeros(n)
    
    # Duration normalized
    max_dur = float(np.max(durations)) if n > 0 else 0.0
    if max_dur > 0:
        norm_dur = durations / max_dur
    else:
        norm_dur = np.zeros(n)
    
    # Composite priority score:
    # -slack (dominant): lower slack = higher priority
    # + resource demand intensity: activities that consume more of a scarce resource
    # + tail length: activities on paths close to the end
    # + small duration tiebreaker
    #
    # Weights chosen so slack dominates but resource contention is captured.
    scores = (-norm_slack * 1.0) + (norm_util * 0.35) + (norm_tail * 0.25) + (norm_dur * 0.05)
    
    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
