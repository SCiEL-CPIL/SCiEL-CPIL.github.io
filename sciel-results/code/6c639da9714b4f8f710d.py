"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def _build_graph(predecessors, n):
    """Build successor list and in-degree from predecessors list."""
    successors = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    return successors, in_degree


def _compute_ranks(instance):
    """
    Compute multiple ranking signals for each task:
    - longest path from start (forward rank)
    - longest path to end (backward rank / urgency)
    - number of transitive descendants (downstream subtree size)
    - number of transitive predecessors (upstream constraint)
    """
    n = len(instance["times"])
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    
    successors, in_degree = _build_graph(predecessors, n)
    
    # Topological order (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # Forward rank: longest path from any source to node i
    forward_rank = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in successors[u]:
            candidate = forward_rank[u] + times[u]
            if candidate > forward_rank[v]:
                forward_rank[v] = candidate
    
    # Backward rank: longest path from node i to any sink
    # Process in reverse topological order
    backward_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        for v in successors[u]:
            candidate = times[u] + backward_rank[v]
            if candidate > backward_rank[u]:
                backward_rank[u] = candidate
    
    # Number of transitive descendants (BFS/DFS count)
    desc_count = np.zeros(n, dtype=int)
    for i in range(n):
        # BFS from i counting all reachable nodes
        visited = set()
        q = deque([i])
        visited.add(i)
        while q:
            u = q.popleft()
            for v in successors[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        desc_count[i] = len(visited) - 1  # exclude self
    
    # Number of transitive predecessors
    pred_count = np.zeros(n, dtype=int)
    # Build reverse graph
    pred_graph = defaultdict(list)
    for i in range(n):
        for p in predecessors[i]:
            pred_graph[i].append(p)
    
    for i in range(n):
        visited = set()
        q = deque([i])
        visited.add(i)
        while q:
            u = q.popleft()
            for p in pred_graph[u]:
                if p not in visited:
                    visited.add(p)
                    q.append(p)
        pred_count[i] = len(visited) - 1
    
    return times, forward_rank, backward_rank, desc_count, pred_count, successors, topo_order


def _greedy_balance(times, priorities, cycle_time, predecessors, successors, topo_order, n):
    """
    Greedy list scheduling: repeatedly pick the highest-priority eligible task
    that fits in the current station. Return number of stations used.
    """
    # eligible: all predecessors already assigned
    assigned = np.zeros(n, dtype=bool)
    # remaining_pred_count[i] = number of unassigned predecessors of i
    rem_pred = np.zeros(n, dtype=int)
    for i in range(n):
        rem_pred[i] = len(predecessors[i])
    
    # Initially eligible tasks are those with rem_pred == 0
    eligible = set(i for i in range(n) if rem_pred[i] == 0)
    
    stations = 0
    station_load = 0.0
    
    while len(assigned) < n:
        # Find best eligible task that fits
        best_task = -1
        best_priority = -np.inf
        for t in eligible:
            if not assigned[t] and times[t] <= (cycle_time - station_load):
                if priorities[t] > best_priority:
                    best_priority = priorities[t]
                    best_task = t
        
        if best_task == -1:
            # No eligible task fits in current station; start new station
            stations += 1
            station_load = 0.0
            # Re-find best eligible task that fits in fresh station
            best_task = -1
            best_priority = -np.inf
            for t in eligible:
                if not assigned[t] and times[t] <= cycle_time:
                    if priorities[t] > best_priority:
                        best_priority = priorities[t]
                        best_task = t
            if best_task == -1:
                # Should not happen if all times <= cycle_time
                # Force assign the highest priority eligible
                for t in eligible:
                    if not assigned[t]:
                        best_task = t
                        break
                if best_task == -1:
                    break
        
        # Assign best_task
        assigned[best_task] = True
        station_load += times[best_task]
        eligible.discard(best_task)
        
        # Update eligible set: successors of best_task may become eligible
        for v in successors[best_task]:
            rem_pred[v] -= 1
            if rem_pred[v] == 0:
                eligible.add(v)
    
    return stations


def _evaluate_priority(times, forward_rank, backward_rank, desc_count, pred_count,
                       successors, topo_order, cycle_time, predecessors, n):
    """
    Evaluate a priority vector by running greedy balancing and return station count.
    """
    return _greedy_balance(times, np.asarray(times, dtype=float) * 0 + forward_rank, cycle_time, predecessors, successors, topo_order, n)


def priority(instance):
    """
    Score each assembly task. The evaluator will use these scores to greedily
    fill stations with the highest-priority eligible task that fits the cycle time.
    
    Strategy: Combine backward rank (urgency), task time, and downstream density
    to produce a priority that places critical-path bottleneck tasks early.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n = len(instance["times"])
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Compute graph properties
    times_arr, forward_rank, backward_rank, desc_count, pred_count, successors, topo_order = _compute_ranks(instance)
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to simple LPT
        return times.copy()
    
    # Design priority:
    # - backward_rank captures how much "work" remains after this task on the critical path
    # - times captures the direct load
    # - desc_count / (backward_rank + 1) gives a density: how many tasks are compressed into the remaining time
    # - We want high priority for tasks that are on critical paths AND have high time
    
    # Normalize components
    max_backward = np.max(backward_rank) if np.max(backward_rank) > 0 else 1.0
    max_time = np.max(times_arr) if np.max(times_arr) > 0 else 1.0
    max_desc = np.max(desc_count) if np.max(desc_count) > 0 else 1.0
    
    # Primary signal: backward rank (urgency)
    # Secondary: task time (LPT component)
    # Tertiary: bottleneck density = time / (1 + normalized backward rank)
    #   This penalizes long tasks that have little slack remaining
    
    # Smooth combination
    # priority = alpha * backward_rank_norm + beta * time_norm + gamma * density
    
    backward_norm = backward_rank / max_backward
    time_norm = times_arr / max_time
    # Density: high when task is long but has few downstream tasks relative to time
    # Use safe division
    density = np.zeros(n, dtype=float)
    for i in range(n):
        br = backward_rank[i]
        if br > 0:
            density[i] = times_arr[i] / (br + 1.0)
        else:
            density[i] = times_arr[i]
    
    max_density = np.max(density) if np.max(density) > 0 else 1.0
    density_norm = density / max_density
    
    # Weighted combination: emphasize backward rank most, then time, then density
    alpha = 0.5
    beta = 0.35
    gamma = 0.15
    
    priorities = alpha * backward_norm + beta * time_norm + gamma * density_norm
    
    # Check time
    if time.time() > deadline - 3:
        return priorities
    
    # Try a few variations and pick the best one via greedy simulation
    best_stations = float('inf')
    best_priorities = priorities.copy()
    
    # Variation 1: Pure backward rank
    p1 = backward_rank.copy()
    # Variation 2: Backward rank * time (product emphasizes both)
    p2 = backward_rank * times_arr
    # Variation 3: Original combination
    p3 = priorities.copy()
    # Variation 4: Forward rank + backward rank
    p4 = forward_rank + backward_rank
    # Variation 5: Backward rank with small LPT tiebreaker
    p5 = backward_rank + 0.01 * times_arr
    
    candidates = [p1, p2, p3, p4, p5]
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 2:
            break
        stations = _greedy_balance(times_arr, cand, cycle_time, predecessors, successors, topo_order, n)
        if stations < best_stations:
            best_stations = stations
            best_priorities = cand.copy()
    
    # Check time
    if time.time() > deadline - 1:
        return best_priorities
    
    # Try a few more refined combinations
    for alpha_val in [0.4, 0.6, 0.7]:
        for beta_val in [0.2, 0.3, 0.4]:
            gamma_val = 1.0 - alpha_val - beta_val
            if gamma_val < 0:
                continue
            if time.time() > deadline - 1:
                break
            p = alpha_val * backward_norm + beta_val * time_norm + gamma_val * density_norm
            stations = _greedy_balance(times_arr, p, cycle_time, predecessors, successors, topo_order, n)
            if stations < best_stations:
                best_stations = stations
                best_priorities = p.copy()
    
    return best_priorities
# EVOLVE-BLOCK-END
