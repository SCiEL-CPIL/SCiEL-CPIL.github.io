"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
from collections import deque, defaultdict


def _build_graph(times, preds):
    n = len(times)
    # successors[i] = list of tasks that depend on i
    successors = [[] for _ in range(n)]
    # predecessors[i] = list of tasks that i depends on (given)
    pred_list = [list(p) for p in preds]
    for j in range(n):
        for i in pred_list[j]:
            successors[i].append(j)
    return pred_list, successors


def _compute_rank(pred_list, successors, times):
    """Compute position rank (how far from end) and downstream work."""
    n = len(times)
    # Compute longest path from each node to any sink (position from start)
    # and total downstream work (sum of times in subtree of successors)
    
    # Topological order
    in_degree = [len(pred_list[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Longest path from start to each node (position)
    pos = [0.0] * n
    for u in topo:
        for v in successors[u]:
            if pos[u] + times[u] > pos[v]:
                pos[v] = pos[u] + times[u]
    
    # Longest path from each node to end (position from end)
    # Process in reverse topo
    pos_end = [0.0] * n
    for u in reversed(topo):
        for v in successors[u]:
            if pos_end[v] + times[v] > pos_end[u]:
                pos_end[u] = pos_end[v] + times[v]
    
    # Total downstream work (including self): sum of times in the subtree of successors + self
    downstream_work = [0.0] * n
    for u in reversed(topo):
        total = times[u]
        for v in successors[u]:
            total += downstream_work[v]
        downstream_work[u] = total
    
    return pos, pos_end, downstream_work


def _greedy_assign(priority_scores, times, pred_list, successors, cycle_time):
    """
    Greedy station assignment:
    - At each step, find all eligible tasks (all predecessors assigned, not yet assigned)
    - Among eligible tasks that fit in remaining station time, pick highest priority
    - If none fit, start new station
    """
    n = len(times)
    assigned = [False] * n
    remaining_preds = [len(pred_list[i]) for i in range(n)]
    
    # Min-heap for priority: use negative priority so smallest (most negative) = highest priority
    # We'll maintain a list of eligible tasks and pick best each time
    
    station_count = 0
    current_station_time = 0.0
    
    # Track eligible tasks
    eligible = set()
    for i in range(n):
        if remaining_preds[i] == 0:
            eligible.add(i)
    
    while len(assigned) - sum(assigned) > 0:
        if not eligible:
            # Should not happen if graph is valid
            break
        
        # Find best eligible task that fits
        best_task = None
        best_prio = -float('inf')
        for t in eligible:
            if times[t] <= cycle_time - current_station_time:
                p = priority_scores[t]
                if p > best_prio:
                    best_prio = p
                    best_task = t
        
        if best_task is None:
            # No eligible task fits; start new station
            station_count += 1
            current_station_time = 0.0
            # Try again
            best_task = None
            best_prio = -float('inf')
            for t in eligible:
                if times[t] <= cycle_time:
                    p = priority_scores[t]
                    if p > best_prio:
                        best_prio = p
                        best_task = t
        
        if best_task is None:
            # A task is longer than cycle time; assign it alone
            # Find any eligible task
            best_task = next(iter(eligible))
            if current_station_time > 0:
                station_count += 1
                current_station_time = 0.0
            # Assign it (it may exceed cycle time, but we must place it)
        
        # Assign best_task
        assigned[best_task] = True
        eligible.discard(best_task)
        current_station_time += times[best_task]
        
        # Update successors
        for v in successors[best_task]:
            remaining_preds[v] -= 1
            if remaining_preds[v] == 0:
                eligible.add(v)
    
    # Count the last station if it has any work
    if current_station_time > 0:
        station_count += 1
    
    return station_count


def _compute_density_priority(times, pred_list, successors):
    """Work Content Density: ratio of task time to downstream subtree work."""
    n = len(times)
    pos, pos_end, downstream_work = _compute_rank(pred_list, successors, times)
    
    # Density = times[i] / downstream_work[i]
    # Higher density means the task is a large fraction of its downstream work
    # We want to schedule high-density tasks first
    density = np.zeros(n)
    for i in range(n):
        if downstream_work[i] > 0:
            density[i] = times[i] / downstream_work[i]
        else:
            density[i] = 0.0
    
    # Combine with position (earlier in critical path should be higher priority)
    # Normalize pos to [0, 1]
    max_pos = max(pos_end) if max(pos_end) > 0 else 1.0
    normalized_pos_end = [pos_end[i] / max_pos for i in range(n)]
    
    # Priority = density * (1 + normalized_pos_end)
    # This favors tasks that are both dense and far from the end
    priority = np.zeros(n)
    for i in range(n):
        priority[i] = density[i] * (1.0 + normalized_pos_end[i])
    
    return priority


def _compute_combined_priority(times, pred_list, successors, cycle_time):
    """
    Compute a combined priority that blends:
    - Work content density
    - Critical path position (pos_end)
    - Number of successors (branching factor)
    """
    n = len(times)
    pos, pos_end, downstream_work = _compute_rank(pred_list, successors, times)
    
    # Density
    density = np.zeros(n)
    for i in range(n):
        if downstream_work[i] > 0:
            density[i] = times[i] / downstream_work[i]
        else:
            density[i] = 0.0
    
    # Normalize components to [0, 1]
    max_density = max(density) if max(density) > 0 else 1.0
    norm_density = density / max_density
    
    max_pos_end = max(pos_end) if max(pos_end) > 0 else 1.0
    norm_pos_end = np.array([pos_end[i] / max_pos_end for i in range(n)])
    
    # Number of immediate successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = max(succ_count) if max(succ_count) > 0 else 1.0
    norm_succ = succ_count / max_succ
    
    # Task time normalized
    max_time = max(times) if max(times) > 0 else 1.0
    norm_time = np.array([times[i] / max_time for i in range(n)])
    
    # Combined priority: weighted sum
    # Emphasize density and position
    priority = 0.4 * norm_density + 0.3 * norm_pos_end + 0.15 * norm_time + 0.15 * norm_succ
    
    return priority


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority eligible task
    that fits the cycle time. Scored by mean number of stations on five generated 30-task
    instances with cycle time 40 (lower is better).
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["preds"] if "preds" in instance else instance.get("predecessors", [])
    cycle_time = instance.get("cycle_time", 40)
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    pred_list, successors = _build_graph(times, preds)
    
    # Compute combined priority
    prio = _compute_combined_priority(times, pred_list, successors, cycle_time)
    
    # Also try density-only priority and pick better via quick simulation
    prio_density = _compute_density_priority(times, pred_list, successors)
    
    # Evaluate both on this instance and pick the one with fewer stations
    stations_a = _greedy_assign(prio, times, pred_list, successors, cycle_time)
    stations_b = _greedy_assign(prio_density, times, pred_list, successors, cycle_time)
    
    if stations_b < stations_a:
        return prio_density
    else:
        return prio
    
    # If we had more time, we could try more variants, but keep it simple for now
    # Return the best found
# EVOLVE-BLOCK-END
