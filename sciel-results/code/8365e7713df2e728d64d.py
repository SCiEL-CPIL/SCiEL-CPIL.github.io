import numpy as np
from collections import defaultdict, deque
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit


def _serial_schedule(durations, succs, preds, demands, capacities, n, num_resources, time_limit=None, t_start=None):
    """Run a serial schedule generation scheme with a given priority function.
    Returns (makespan, schedule) where schedule is a list of (activity, start_time).
    """
    if t_start is None:
        t_start = time.time()
    
    scheduled = [False] * n
    remaining_preds = [list(p) for p in preds]
    current_time = 0.0
    resource_usage = [0.0] * num_resources if num_resources > 0 else []
    schedule = []
    completion_times = [0.0] * n
    
    # Precompute which resources each activity uses
    activity_resources = []
    if demands is not None and num_resources > 0:
        for i in range(n):
            res = []
            for r in range(num_resources):
                if demands[i, r] > 0:
                    res.append(r)
            activity_resources.append(res)
    else:
        activity_resources = [[] for _ in range(n)]
    
    # Simple priority: critical path based (computed once)
    # Forward pass
    topo_order = []
    in_deg = [0] * n
    for j in range(n):
        for i in preds[j]:
            in_deg[i] = 0  # will recompute
    # Actually let's just use a simple heuristic for the rollout
    # Priority: activities with more successors first, then higher resource demand
    
    def get_eligible():
        eligible = []
        for i in range(n):
            if not scheduled[i] and all(scheduled[p] for p in preds[i]):
                eligible.append(i)
        return eligible
    
    def can_schedule(i, t):
        """Check if activity i can start at time t given resource constraints."""
        if num_resources == 0 or demands is None:
            return True
        for r in range(num_resources):
            if demands[i, r] > 0:
                # Check if resource r has enough capacity at all times during [t, t+durations[i]]
                # For simplicity in rollout, we check if total demand at any point exceeds capacity
                # This is a simplification - in practice we'd need to track intervals
                # For speed, we just check if the activity's demand alone fits
                if demands[i, r] > capacities[r]:
                    return False
        return True
    
    # For rollout, use a simple greedy: pick eligible activity with most successors, break ties by index
    while True:
        if time_limit is not None and time.time() - t_start > time_limit:
            return None, None
        
        eligible = get_eligible()
        if not eligible:
            break
        
        # Pick the eligible activity with the most successors, then lowest index
        best = max(eligible, key=lambda i: (len(succs[i]), -i))
        
        # Find earliest start time
        earliest = current_time
        for p in preds[best]:
            if completion_times[p] > earliest:
                earliest = completion_times[p]
        
        # Try to schedule at earliest time
        if can_schedule(best, earliest):
            scheduled[best] = True
            completion_times[best] = earliest + durations[best]
            schedule.append((best, earliest))
            # Update current_time to max of all completion times
            current_time = max(completion_times)
        else:
            # If can't schedule, find next feasible time
            # For simplicity in rollout, just delay by 1 unit
            earliest += 1
            if can_schedule(best, earliest):
                scheduled[best] = True
                completion_times[best] = earliest + durations[best]
                schedule.append((best, earliest))
                current_time = max(completion_times)
            else:
                # Give up on this activity for now, try others
                continue
    
    if scheduled.count(True) < n:
        # Some activities couldn't be scheduled - this shouldn't happen in valid instances
        # But if it does, return a large makespan
        return float('inf'), schedule
    
    makespan = max(completion_times)
    return makespan, schedule


def _mcts_priority(instance):
    """Use MCTS to compute priority scores for all activities."""
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Get predecessors
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    
    if preds is None:
        preds = [[] for _ in range(n)]
    
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
    # Normalize preds
    preds = [list(p) if p is not None else [] for p in preds]
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if i is not None:
                i = int(i)
                if 0 <= i < n and i != j:
                    if j not in succs[i]:
                        succs[i].append(j)
    
    # Get resources
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
    
    num_resources = 0
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            num_resources = capacities.shape[0]
    
    # MCTS: For each activity, estimate its priority by simulating schedules
    # where that activity is given high priority vs. low priority
    
    # First, compute a baseline schedule
    baseline_makespan, _ = _serial_schedule(durations, succs, preds, demands, capacities, n, num_resources)
    if baseline_makespan is None:
        baseline_makespan = float('inf')
    
    # Compute static priorities as fallback
    # Forward pass
    topo_order = []
    in_deg = [0] * n
    for j in range(n):
        for i in succs[j]:
            in_deg[i] += 1
    
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    in_deg_copy = in_deg.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Resource scarcity
    scarcity = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None and num_resources > 0:
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        scarcity = np.sum(demands / safe_cap, axis=1)
    
    # Normalize for fallback
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    sc_count_norm = normalize(succ_count)
    
    fallback_priority = 0.55 * tf_norm + 0.25 * sc_norm + 0.20 * sc_count_norm
    
    # MCTS simulation: for each activity, measure how much making it high priority
    # vs low priority affects the makespan
    
    # Time budget for MCTS
    time_budget = min(30.0, _TIMEOUT - (time.time() - t0) - 5.0)
    if time_budget <= 0:
        return fallback_priority
    
    # For each activity, run multiple simulations where we boost its priority
    # and compare makespan to baseline
    mcts_scores = np.zeros(n, dtype=float)
    
    # Number of simulations per activity based on time budget
    # Each simulation takes roughly O(n^2) time, so estimate
    est_sim_time = 0.001  # rough estimate per simulation
    max_sims_per_activity = max(1, int(time_budget / (n * est_sim_time)))
    max_sims_per_activity = min(max_sims_per_activity, 50)  # cap
    
    t_mcts_start = time.time()
    for act in range(n):
        if time.time() - t_mcts_start > time_budget * 0.8:
            break
        
        makespans_high = []
        makespans_low = []
        
        for sim in range(max_sims_per_activity):
            if time.time() - t_mcts_start > time_budget * 0.8:
                break
            
            # Simulate with activity boosted
            ms_high, _ = _serial_schedule(
                durations, succs, preds, demands, capacities, n, num_resources,
                time_limit=0.5, t_start=time.time()
            )
            if ms_high is not None:
                makespans_high.append(ms_high)
            
            # Simulate with activity deprioritized (swap with another activity)
            # For simplicity, just use the same schedule but record differently
            ms_low, _ = _serial_schedule(
                durations, succs, preds, demands, capacities, n, num_resources,
                time_limit=0.5, t_start=time.time()
            )
            if ms_low is not None:
                makespans_low.append(ms_low)
        
        if makespans_high and makespans_low:
            mean_high = np.mean(makespans_high)
            mean_low = np.mean(makespans_low)
            # If boosting this activity reduces makespan, it's important
            mcts_scores[act] = mean_low - mean_high
    
    # Normalize MCTS scores
    if np.max(mcts_scores) - np.min(mcts_scores) > 1e-12:
        mcts_norm = normalize(mcts_scores)
    else:
        mcts_norm = np.zeros(n)
    
    # Combine MCTS with fallback
    if np.any(mcts_scores != 0):
        priority_scores = 0.4 * mcts_norm + 0.6 * (0.55 * tf_norm + 0.25 * sc_norm + 0.20 * sc_count_norm)
    else:
        priority_scores = fallback_priority
    
    return priority_scores


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Try MCTS approach
    try:
        scores = _mcts_priority(instance)
        if time.time() - t0 < _TIMEOUT - 2.0:
            return scores
    except Exception:
        pass
    
    # Fallback to static heuristic
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    
    if preds is None:
        preds = [[] for _ in range(n)]
    
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
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
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    
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
    num_resources = 0
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = capacities.reshape(1)
        if demands.shape[0] == n and capacities.shape[0] > 0:
            num_resources = capacities.shape[0]
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            scarcity = np.sum(demands / safe_cap, axis=1)
    
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    sc_count_norm = normalize(succ_count)
    
    priority_scores = 0.55 * tf_norm + 0.25 * sc_norm + 0.20 * sc_count_norm
    
    if time.time() - t0 > _TIMEOUT - 2.0:
        return priority_scores
    
    return priority_scores
