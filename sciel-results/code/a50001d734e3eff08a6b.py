import numpy as np
from collections import defaultdict, deque
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    
    if preds is None:
        preds = [[] for _ in range(n)]
    
    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        if j not in succs[i]:
                            succs[i].append(j)
    
    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo_order, there's a cycle; add remaining
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Forward pass: ES (Earliest Start)
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0.0
    
    # Backward pass: LS (Latest Start)
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    
    # Resource scarcity
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break
    
    scarcity = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            scarcity = np.sum(demands / safe_cap, axis=1)
    
    # Successor count (number of immediate successors)
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Depth: number of activities that must complete before this one can start
    # (i.e., ES value normalized)
    depth = ES.copy()
    
    # Breadth: number of successors including transitive ones (approximated by out-degree in transitive closure)
    # For efficiency, use just immediate successor count + depth as proxy
    
    # Weighted tardiness proxy: activities with later ES but small float are critical
    # Use a combined score that emphasizes:
    # 1. Low total float (critical path) - most important
    # 2. High resource consumption (scarce resource users first)
    # 3. Early start (to avoid delaying downstream)
    # 4. High successor count (bottleneck activities)
    
    # Normalize each component to [0, 1]
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Inverse total float: lower float -> higher priority
    inv_tf = normalize(-total_float)
    
    # Resource scarcity: higher demand/capacity ratio -> higher priority
    sc_norm = normalize(scarcity)
    
    # Successor count: more successors -> higher priority (bottleneck)
    succ_norm = normalize(succ_count)
    
    # Depth: earlier start -> lower priority for depth, but we want to schedule
    # critical activities early. Use negative depth so earlier ES gets lower score
    # Actually, for serial SGS, we want to pick activities that are "ready" early
    # and on critical path. Let's use a different approach.
    
    # Key insight: In serial SGS, at each step we pick from eligible activities.
    # The best heuristic prioritizes:
    # - Activities on or near the critical path (low total float)
    # - Activities that consume scarce resources (to resolve conflicts early)
    # - Activities with many successors (to unlock more of the network)
    
    # Use adaptive weights based on instance characteristics
    # If resources are very tight, increase scarcity weight
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        caps_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(-1, 1)
        if caps_arr.ndim == 0:
            caps_arr = caps_arr.reshape(1)
        if demands_arr.shape[0] == n and caps_arr.shape[0] > 0:
            # Average utilization ratio
            avg_util = np.mean(demands_arr / np.where(caps_arr > 0, caps_arr, 1.0))
            # Higher utilization -> more resource conflict -> higher scarcity weight
            w_sc = min(0.35, 0.10 + 0.25 * avg_util)
        else:
            w_sc = 0.15
    else:
        w_sc = 0.10
    
    # Weights: total float dominates
    w_tf = 0.65
    w_sc = 0.25 if demands is not None else 0.15
    w_succ = 0.10 if demands is not None else 0.20
    
    # Adjust so weights sum to 1
    total_w = w_tf + w_sc + w_succ
    w_tf /= total_w
    w_sc /= total_w
    w_succ /= total_w
    
    priority_scores = w_tf * inv_tf + w_sc * sc_norm + w_succ * succ_norm
    
    # Deadline guard
    if time.time() - t0 > 10.0:
        return priority_scores
    
    return priority_scores
