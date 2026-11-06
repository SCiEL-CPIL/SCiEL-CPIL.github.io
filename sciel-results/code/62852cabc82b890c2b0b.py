"""Resource-Constrained Project Scheduling: priority scoring for serial SGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance, n):
    """Build predecessor/successor lists and resource demand matrix."""
    preds_raw = instance.get("predecessors", None)
    if preds_raw is None:
        preds_raw = [[] for _ in range(n)]
    
    preds = []
    succs = [[] for _ in range(n)]
    for i in range(n):
        p = preds_raw[i]
        if p is None:
            p = []
        elif isinstance(p, (int, np.integer)):
            p = [int(p)]
        else:
            p = [int(x) for x in p]
        preds.append(p)
        for pi in p:
            succs[pi].append(i)
    
    # Resource demands: instance["demands"] is shape (n, R) or list of lists
    demands = np.asarray(instance["demands"], dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
    capacities = np.asarray(instance["capacities"], dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(demands.shape[1], float(capacities))
    
    return preds, succs, demands, capacities


def _compute_cpm_priorities(durations, preds, succs, n):
    """
    Compute CPM-based priority scores:
    - ES (earliest start) via forward pass
    - LF (latest finish) via backward pass
    - Total float = LF - ES - duration
    - Priority = -float (lower float = higher priority)
    Also compute resource scarcity pressure.
    """
    durations = np.asarray(durations, dtype=float)
    
    # Forward pass: ES[i] = max(ES[p] + dur[p]) for all preds p
    ES = np.zeros(n)
    # Topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    from collections import deque
    q = deque(queue)
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    # Forward pass
    for u in topo_order:
        for v in succs[u]:
            if ES[v] < ES[u] + durations[u]:
                ES[v] = ES[u] + durations[u]
    
    # Backward pass: LF[i] = min(LF[s] - dur[s]) for all succs s
    # Project end time
    project_end = np.max(ES + durations) if n > 0 else 0.0
    LF = np.full(n, project_end)
    
    for u in reversed(topo_order):
        for v in succs[u]:
            if LF[u] > LF[v] - durations[v]:
                LF[u] = LF[v] - durations[v]
    
    # Total float
    TF = LF - ES - durations
    # Priority: negative float (activities with less slack are more critical)
    cpm_priority = -TF
    
    return cpm_priority, ES, LF


def _resource_pressure(demands, capacities, n):
    """
    Compute resource scarcity pressure for each activity.
    An activity using a scarce resource (high demand/capacity ratio) gets higher priority.
    """
    R = demands.shape[1] if demands.ndim > 1 else 0
    if R == 0:
        return np.zeros(n)
    
    # For each resource, compute max demand across activities
    # Pressure for activity i = sum over resources r of (demand[i,r] / capacity[r]) * scarcity_r
    # where scarcity_r = max_demand_r / capacity_r (how tight the resource is)
    
    max_demand_per_res = np.max(demands, axis=0)  # shape (R,)
    scarcity = max_demand_per_res / np.maximum(capacities, 1e-9)  # shape (R,)
    
    # Normalize scarcity to [0, 1]
    max_scarcity = np.max(scarcity) if len(scarcity) > 0 else 1.0
    if max_scarcity > 0:
        scarcity_norm = scarcity / max_scarcity
    else:
        scarcity_norm = scarcity
    
    # Demand ratio for each activity-resource pair
    demand_ratio = demands / np.maximum(capacities, 1e-9)  # shape (n, R)
    
    # Weighted sum: how much scarce resources does this activity use?
    pressure = np.sum(demand_ratio * scarcity_norm, axis=1)
    
    return pressure


def _most_preceding_successors(preds, succs, n):
    """
    MPS rule: number of immediate successors (direct dependents).
    Activities with more successors should be scheduled earlier.
    """
    mps = np.zeros(n)
    for i in range(n):
        mps[i] = len(succs[i])
    return mps


def _successor_count(preds, succs, n):
    """
    Count total number of successors (transitive) for each activity.
    Activities with more downstream work should be prioritized.
    """
    # Use topological order to compute transitive successor counts
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # succ_count[i] = number of transitive successors of i
    succ_count = np.zeros(n)
    for u in reversed(topo_order):
        succ_count[u] = len(succs[u])
        for v in succs[u]:
            succ_count[u] += succ_count[v]
    
    return succ_count


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (scheduled earlier when eligible).
    
    Combines:
    1. CPM total float (critical path activities first)
    2. Resource scarcity pressure (activities using tight resources first)
    3. Most Preceding Successors (activities with more dependents first)
    4. Duration (longer activities first)
    5. Transitive successor count (activities with more downstream work first)
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
    
    preds, succs, demands, capacities = _build_graph(instance, n)
    
    # Component 1: CPM priority (negative total float)
    cpm_prio, ES, LF = _compute_cpm_priorities(durations, preds, succs, n)
    
    # Normalize CPM priority to [0, 1]
    cpm_min = np.min(cpm_prio)
    cpm_max = np.max(cpm_prio)
    cpm_range = cpm_max - cpm_min
    if cpm_range > 1e-9:
        cpm_norm = (cpm_prio - cpm_min) / cpm_range
    else:
        cpm_norm = np.zeros(n)
    
    # Component 2: Resource pressure
    pressure = _resource_pressure(demands, capacities, n)
    pr_min = np.min(pressure)
    pr_max = np.max(pressure)
    pr_range = pr_max - pr_min
    if pr_range > 1e-9:
        pr_norm = (pressure - pr_min) / pr_range
    else:
        pr_norm = np.zeros(n)
    
    # Component 3: MPS (immediate successors)
    mps = _most_preceding_successors(preds, succs, n)
    mps_min = np.min(mps)
    mps_max = np.max(mps)
    mps_range = mps_max - mps_min
    if mps_range > 1e-9:
        mps_norm = (mps - mps_min) / mps_range
    else:
        mps_norm = np.zeros(n)
    
    # Component 4: Duration
    dur_min = np.min(durations)
    dur_max = np.max(durations)
    dur_range = dur_max - dur_min
    if dur_range > 1e-9:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.zeros(n)
    
    # Component 5: Transitive successor count
    succ_count = _successor_count(preds, succs, n)
    sc_min = np.min(succ_count)
    sc_max = np.max(succ_count)
    sc_range = sc_max - sc_min
    if sc_range > 1e-9:
        sc_norm = (succ_count - sc_min) / sc_range
    else:
        sc_norm = np.zeros(n)
    
    # Weighted combination
    # Critical path (float) is most important, then resource pressure, then others
    weights = {
        'cpm': 0.40,
        'pressure': 0.25,
        'mps': 0.15,
        'duration': 0.10,
        'succ_count': 0.10,
    }
    
    score = (weights['cpm'] * cpm_norm +
             weights['pressure'] * pr_norm +
             weights['mps'] * mps_norm +
             weights['duration'] * dur_norm +
             weights['succ_count'] * sc_norm)
    
    return score
# EVOLVE-BLOCK-END
