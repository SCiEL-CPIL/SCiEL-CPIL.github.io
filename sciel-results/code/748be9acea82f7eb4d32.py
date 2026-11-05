"""
Strategy: Dynamic Bottleneck Critical Ratio
Instead of static longest path, we compute a 'critical ratio' for each task.
The ratio is (Sum of times of all successors + own time) / own time.
This identifies tasks that act as high-volume bottlenecks relative to their duration.
Prioritizing these tasks helps prevent idle time in subsequent stations by ensuring
that tasks with heavy downstream dependencies are started early enough to keep the line moving.

We also include a lightweight local search to refine the priority vector if time permits,
but the core structural change is the new scoring metric.
"""
import numpy as np
import time
from collections import deque

def _compute_downstream_workload(times, predecessors, n):
    """
    Compute the total workload (sum of times) for each task and all its successors.
    This is the 'remaining downstream workload' including the task itself.
    """
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Topological sort to process in reverse order (sinks first)
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If not all nodes were processed, there's a cycle. Handle gracefully by filling missing.
    if len(topo_order) < n:
        # Fallback: just use times for any unprocessed nodes (should be rare/invalid)
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)
                
    # Reverse topological order to calculate downstream sums
    downstream_sum = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        downstream_sum[u] = times[u]
        for v in successors[u]:
            downstream_sum[u] += downstream_sum[v]
            
    return downstream_sum

def _simulate_greedy(priority_scores, times, predecessors, cycle_time):
    """
    Simulates the greedy assignment process.
    Returns the number of stations used.
    """
    n = len(times)
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_tasks = list(range(n))
    
    # Pre-compute successor sets for faster eligibility checks? 
    # For N=30, simple checks are fine, but let's optimize slightly.
    
    while remaining_tasks:
        best_task_idx = -1
        best_priority = -np.inf
        
        # Find the highest priority task that is eligible and fits
        for i in remaining_tasks:
            if assigned[i]:
                continue
            
            # Check eligibility: all predecessors must be assigned
            eligible = True
            for p in predecessors[i]:
                if not assigned[p]:
                    eligible = False
                    break
            
            if not eligible:
                continue
            
            # Check if it fits in the current station
            if station_load + times[i] <= cycle_time:
                if priority_scores[i] > best_priority:
                    best_priority = priority_scores[i]
                    best_task_idx = i
        
        if best_task_idx != -1:
            # Assign the task
            assigned[best_task_idx] = True
            station_load += times[best_task_idx]
            remaining_tasks.remove(best_task_idx)
        else:
            # No task fits in the current station. Start a new station.
            num_stations += 1
            station_load = 0.0
            
            # In a new station, find the highest priority eligible task that fits (time <= cycle_time)
            best_task_idx = -1
            best_priority = -np.inf
            
            for i in remaining_tasks:
                if assigned[i]:
                    continue
                
                eligible = True
                for p in predecessors[i]:
                    if not assigned[p]:
                        eligible = False
                        break
                
                if not eligible:
                    continue
                
                if times[i] <= cycle_time:
                    if priority_scores[i] > best_priority:
                        best_priority = priority_scores[i]
                        best_task_idx = i
            
            if best_task_idx != -1:
                assigned[best_task_idx] = True
                station_load += times[best_task_idx]
                remaining_tasks.remove(best_task_idx)
            else:
                # Should not happen in feasible instances
                break
                
    return num_stations

def priority(instance):
    """
    Computes priority scores based on the Critical Ratio heuristic.
    Score = Downstream_Workload / Own_Time
    Higher ratio means the task is a bigger bottleneck relative to its size, so it should be prioritized.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit
    
    # 1. Compute downstream workload
    downstream_workload = _compute_downstream_workload(times, predecessors, n)
    
    # 2. Compute Critical Ratio
    # Avoid division by zero
    safe_times = np.where(times == 0, 1e-9, times)
    critical_ratio = downstream_workload / safe_times
    
    # 3. Base priority is the critical ratio
    best_priority_scores = critical_ratio.copy()
    best_stations = _simulate_greedy(best_priority_scores, times, predecessors, cycle_time)
    
    # 4. Local Search / Refinement
    # We can try to improve the score by slightly perturbing the ratios to break ties or adjust order.
    # Since the critical ratio is a strong structural signal, we use very small noise.
    
    rng = np.random.default_rng(42)
    
    # Try a few variations of noise scaling
    noise_scales = [0.0, 0.01, 0.05, 0.1]
    
    for scale in noise_scales:
        if time.time() > deadline - 5.0:
            break
            
        if scale == 0.0:
            candidate_scores = best_priority_scores.copy()
        else:
            # Add small noise to break ties or slightly adjust
            noise = rng.uniform(-scale, scale, n)
            candidate_scores = best_priority_scores + noise
            
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            # Check lower bound
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break
                
    # If time remains, do a few random restarts with different noise seeds
    # This is a structural addition: using randomness to explore the neighborhood of the critical ratio.
    iterations = 0
    max_iterations = 50
    while time.time() < deadline - 2.0 and iterations < max_iterations:
        iterations += 1
        # Random perturbation
        noise = rng.uniform(-0.1, 0.1, n)
        candidate_scores = best_priority_scores + noise
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                break

    return best_priority_scores
