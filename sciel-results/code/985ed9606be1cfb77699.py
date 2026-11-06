"""Assembly line balancing with downstream-weight and tightness priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_graph(predecessors, n):
    """Build successor lists and compute topological order."""
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    # Kahn's algorithm for topological order
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1  # Wait, predecessors[i] are tasks that must come before i
    # Actually, in_degree[i] = len(predecessors[i])
    in_degree = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    # If topo is incomplete, there's a cycle; handle gracefully
    if len(topo) < n:
        # Add remaining nodes in arbitrary order to avoid crash
        remaining = set(range(n)) - set(topo)
        topo.extend(sorted(remaining))
    return successors, topo

def _compute_downstream_weight(times, successors, topo):
    """Compute total time of each task plus all its descendants (downstream subtree weight)."""
    n = len(times)
    down_weight = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for i in reversed(range(n)):
        u = topo[i]
        down_weight[u] = times[u]
        for v in successors[u]:
            down_weight[u] += down_weight[v]
    return down_weight

def _compute_successor_count(successors, n):
    """Count immediate successors for each task."""
    return np.array([len(successors[i]) for i in range(n)], dtype=float)

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Simulate the greedy station-filling process to count stations.
    Stations are filled with the highest-priority eligible task that fits the cycle time.
    """
    n = len(times)
    if n == 0:
        return 0
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Track remaining predecessors count for each task
    rem_preds = [len(predecessors[i]) for i in range(n)]
    # Track which tasks are done
    done = [False] * n
    # Track which tasks are eligible (all predecessors done)
    eligible = [rem_preds[i] == 0 for i in range(n)]
    
    # We need to simulate: repeatedly pick the highest-priority eligible task that fits
    # "Fits" means: current station has enough remaining cycle time for this task
    # A task is eligible if all its predecessors are done AND it fits in the current station
    
    # Actually, re-reading: "stations are filled with the highest-priority eligible task that fits the cycle time"
    # This suggests: for each station, we greedily add tasks. A task is eligible if its predecessors
    # are all assigned to earlier stations (or this station? No, predecessors must be in earlier stations
    # or at least completed). Let's assume predecessors must be in strictly earlier stations.
    
    # Let me re-interpret: We assign tasks to stations one by one. For each station, we select
    # the highest-priority task among those whose predecessors are all in previous stations,
    # and that fits in the remaining capacity of the current station. We keep adding tasks to
    # the current station until no eligible task fits. Then we start a new station.
    
    # Actually, a more standard interpretation: for each station, we pick the highest-priority
    # eligible task that fits, assign it, and repeat until no eligible task fits in the remaining
    # capacity. Then move to next station.
    
    # Let's implement this:
    station_count = 0
    assigned = [False] * n
    # Predecessors must be in strictly earlier stations, so a task is eligible for station s
    # if all its predecessors are assigned to stations < s.
    
    # We'll track the station number each task is assigned to
    station_of_task = [-1] * n
    
    # For efficiency with n=30, we can do a simple O(n^2) approach
    remaining_tasks = set(range(n))
    
    while remaining_tasks:
        station_count += 1
        current_station_load = 0.0
        # For this station, repeatedly pick the best eligible task that fits
        while True:
            best_task = -1
            best_priority = -np.inf
            for t in remaining_tasks:
                if assigned[t]:
                    continue
                # Check if all predecessors are assigned to earlier stations
                pred_ok = True
                for p in predecessors[t]:
                    if station_of_task[p] == -1:  # Not assigned yet
                        pred_ok = False
                        break
                    if station_of_task[p] == station_count:  # Same station, not allowed
                        pred_ok = False
                        break
                if not pred_ok:
                    continue
                # Check if it fits
                if current_station_load + times[t] > cycle_time + 1e-9:
                    continue
                if priority_scores[t] > best_priority:
                    best_priority = priority_scores[t]
                    best_task = t
            if best_task == -1:
                break
            # Assign best_task to this station
            assigned[best_task] = True
            station_of_task[best_task] = station_count
            current_station_load += times[best_task]
            remaining_tasks.discard(best_task)
    
    return station_count

def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    deadline = start_time + 145.0  # 5s safety margin
    
    # Build graph
    successors, topo = _build_graph(predecessors, n)
    
    # Compute downstream weights
    down_weight = _compute_downstream_weight(times, successors, topo)
    
    # Compute successor count
    succ_count = _compute_successor_count(successors, n)
    
    # Compute slack
    slack = cycle_time - times
    
    # Combine features:
    # 1. Downstream weight (higher is better - place high-impact tasks early)
    # 2. Slack (lower is better - place tight tasks early)  
    # 3. Successor count (more successors = more branching, can be good or bad)
    
    # Normalize features
    dw_min, dw_max = down_weight.min(), down_weight.max()
    dw_range = dw_max - dw_min if dw_max > dw_min else 1.0
    dw_norm = (down_weight - dw_min) / dw_range
    
    slack_min, slack_max = slack.min(), slack.max()
    slack_range = slack_max - slack_min if slack_max > slack_min else 1.0
    slack_norm = (slack - slack_min) / slack_range  # Lower slack = lower norm
    
    succ_min, succ_max = succ_count.min(), succ_count.max()
    succ_range = succ_max - succ_min if succ_max > succ_min else 1.0
    succ_norm = (succ_count - succ_min) / succ_range
    
    # Priority: high downstream weight, low slack, moderate successor count
    # We want to maximize this score
    # Weight: downstream weight is most important, then low slack, then successor count
    # Let's try: priority = dw_norm * (1.0 - slack_norm) * (1.0 + 0.5 * succ_norm)
    
    priority_scores = dw_norm * (1.0 - slack_norm) * (1.0 + 0.5 * succ_norm)
    
    # Check time
    if time.time() > deadline:
        return priority_scores
    
    # Try a few variations and pick the one that gives the best (lowest) station count
    # via greedy simulation
    best_priority = priority_scores
    best_stations = _greedy_balance(times, predecessors, cycle_time, priority_scores)
    
    # Try variation 1: emphasize downstream weight more
    priority_v1 = dw_norm * 2.0 * (1.0 - slack_norm)
    if time.time() < deadline:
        stations_v1 = _greedy_balance(times, predecessors, cycle_time, priority_v1)
        if stations_v1 < best_stations:
            best_stations = stations_v1
            best_priority = priority_v1
    
    # Try variation 2: emphasize low slack more
    priority_v2 = (1.0 - slack_norm) * 2.0 * dw_norm
    if time.time() < deadline:
        stations_v2 = _greedy_balance(times, predecessors, cycle_time, priority_v2)
        if stations_v2 < best_stations:
            best_stations = stations_v2
            best_priority = priority_v2
    
    # Try variation 3: pure downstream weight
    if time.time() < deadline:
        stations_v3 = _greedy_balance(times, predecessors, cycle_time, down_weight)
        if stations_v3 < best_stations:
            best_stations = stations_v3
            best_priority = down_weight.copy()
    
    # Try variation 4: time-based (longest first)
    if time.time() < deadline:
        stations_v4 = _greedy_balance(times, predecessors, cycle_time, times)
        if stations_v4 < best_stations:
            best_stations = stations_v4
            best_priority = times.copy()
    
    return best_priority
# EVOLVE-BLOCK-END
