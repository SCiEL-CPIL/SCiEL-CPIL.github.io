"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    
    The evaluator fills stations with the highest-priority eligible task that fits
    the cycle time. Lower total number of stations is better.
    
    We use a priority score that combines multiple well-known ALB heuristics:
    1. Longest path in the successor DAG (positional priority)
    2. Task duration (longer tasks first)
    3. Number of immediate successors (tasks with more dependents get higher priority)
    4. Rank position in topological order (earlier = higher priority)
    5. Number of immediate predecessors (tasks with fewer preds can be placed earlier)
    6. A simulated greedy pass to estimate how "critical" each task is
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    # predecessors[i] = list of task indices that must precede task i
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Position in topological order (earlier = lower index = higher priority)
    top_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        top_pos[u] = idx
    
    # Longest path from each node to sink (reverse topological order)
    # dist[i] = longest sum of times from i to any terminal node
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist[v])
        dist[u] = times[u] + max_succ
    
    # Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Number of immediate predecessors
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)
    
    # Total number of successors (transitive)
    total_succ_count = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ_count[v]
        total_succ_count[u] = s
    
    # Deadline guard
    if time.time() - start_time > 80:
        return times
    
    # Simulate the greedy assignment to estimate criticality
    # This gives us a sense of which tasks are "bottlenecks" in practice
    # We do a quick simulation using a simple priority (dist + time) to see
    # which tasks end up in which stations, then use station number as a signal
    
    # Simple greedy simulation
    remaining = set(range(n))
    assigned_station = np.zeros(n, dtype=float)
    station_load = 0.0
    station_num = 0
    pred_count = n_pred.copy()
    
    while remaining:
        # Find eligible tasks (all predecessors assigned)
        eligible = [t for t in remaining if pred_count[t] == 0]
        
        if not eligible:
            # Should not happen in valid DAG, but break to avoid infinite loop
            break
        
        # Sort eligible by (dist + time) descending, then by time descending
        eligible.sort(key=lambda t: (-(dist[t] + times[t]), -times[t]))
        
        # Assign tasks to current station until none fit
        placed_any = False
        for t in eligible:
            if station_load + times[t] <= cycle_time:
                assigned_station[t] = station_num
                station_load += times[t]
                remaining.remove(t)
                placed_any = True
                # Update pred counts for successors
                for v in succ[t]:
                    pred_count[v] -= 1
        
        if not placed_any:
            # No eligible task fits in current station; start new station
            # But we need at least one task per station; the highest-priority
            # eligible task must go somewhere. If none fit, we have a problem.
            # In practice, this shouldn't happen if cycle_time >= max(time).
            # Force place the first eligible task in a new station.
            t = eligible[0]
            assigned_station[t] = station_num
            station_load = times[t]
            remaining.remove(t)
            for v in succ[t]:
                pred_count[v] -= 1
        
        station_num += 1
    
    # Deadline guard
    if time.time() - start_time > 100:
        return times
    
    # Normalize all components
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    norm_dist = normalize(dist)
    norm_times = normalize(times)
    norm_n_succ = normalize(n_succ)
    norm_total_succ = normalize(total_succ_count)
    # Invert top_pos so earlier position = higher priority
    norm_top_pos = 1.0 - normalize(top_pos)
    # Invert n_pred so fewer predecessors = higher priority
    norm_n_pred = 1.0 - normalize(n_pred)
    # Invert assigned_station so earlier station = higher priority
    norm_station = 1.0 - normalize(assigned_station)
    
    # Weighting: combine multiple heuristics
    # Emphasize downstream path, task duration, and simulated station position
    w_dist = 0.25
    w_times = 0.25
    w_n_succ = 0.05
    w_total_succ = 0.05
    w_top_pos = 0.10
    w_n_pred = 0.05
    w_station = 0.25
    
    score = (w_dist * norm_dist + 
             w_times * norm_times + 
             w_n_succ * norm_n_succ + 
             w_total_succ * norm_total_succ +
             w_top_pos * norm_top_pos +
             w_n_pred * norm_n_pred +
             w_station * norm_station)
    
    # Add a small tie-breaker based on original time to break symmetry
    score = score + 1e-6 * times
    
    if time.time() - start_time > 120:
        return times
    
    return score
# EVOLVE-BLOCK-END
