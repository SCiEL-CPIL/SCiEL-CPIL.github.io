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
    Returns: downstream_sum, successors_list, succ_count
    """
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
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
            
    return downstream_sum, successors, succ_count

def _simulate_greedy(priority_scores, times, predecessors, cycle_time):
    """
    Simulates the greedy assignment process.
    Returns the number of stations used.
    Optimized with bitmask for eligibility tracking.
    """
    n = len(times)
    assigned_mask = 0
    station_load = 0.0
    num_stations = 1
    remaining_count = n
    
    # Precompute predecessor bitmasks
    pred_masks = []
    for i in range(n):
        mask = 0
        for p in predecessors[i]:
            mask |= (1 << p)
        pred_masks.append(mask)
    
    # Precompute task times as list for faster access
    t_list = list(times)
    p_list = list(priority_scores)
    
    while remaining_count > 0:
        best_task_idx = -1
        best_priority = -np.inf
        
        # Find the highest priority task that is eligible and fits
        for i in range(n):
            if (assigned_mask >> i) & 1:
                continue
            
            # Check eligibility: all predecessors must be assigned
            if (assigned_mask & pred_masks[i]) != pred_masks[i]:
                continue
            
            # Check if it fits in the current station
            if station_load + t_list[i] <= cycle_time:
                if p_list[i] > best_priority:
                    best_priority = p_list[i]
                    best_task_idx = i
        
        if best_task_idx != -1:
            # Assign the task
            assigned_mask |= (1 << best_task_idx)
            station_load += t_list[best_task_idx]
            remaining_count -= 1
        else:
            # No task fits in the current station. Start a new station.
            num_stations += 1
            station_load = 0.0
            
            # In a new station, find the highest priority eligible task that fits (time <= cycle_time)
            best_task_idx = -1
            best_priority = -np.inf
            
            for i in range(n):
                if (assigned_mask >> i) & 1:
                    continue
                
                eligible = (assigned_mask & pred_masks[i]) == pred_masks[i]
                
                if not eligible:
                    continue
                
                if t_list[i] <= cycle_time:
                    if p_list[i] > best_priority:
                        best_priority = p_list[i]
                        best_task_idx = i
            
            if best_task_idx != -1:
                assigned_mask |= (1 << best_task_idx)
                station_load += t_list[best_task_idx]
                remaining_count -= 1
            else:
                # Should not happen in feasible instances
                break
                
    return num_stations

def _composite_priority(critical_ratio, downstream_workload, succ_count, times, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Composite priority blending critical ratio, downstream workload, and successor count.
    This creates a smoother landscape for local search to navigate.
    """
    n = len(critical_ratio)
    # Normalize each component to [0, 1]
    cr_min, cr_max = np.min(critical_ratio), np.max(critical_ratio)
    dw_min, dw_max = np.min(downstream_workload), np.max(downstream_workload)
    sc_min, sc_max = np.min(succ_count), np.max(succ_count)
    
    if cr_max - cr_min < 1e-12:
        cr_norm = np.ones(n) / n
    else:
        cr_norm = (critical_ratio - cr_min) / (cr_max - cr_min)
    
    if dw_max - dw_min < 1e-12:
        dw_norm = np.ones(n) / n
    else:
        dw_norm = (downstream_workload - dw_min) / (dw_max - dw_min)
    
    if sc_max - sc_min < 1e-12:
        sc_norm = np.ones(n) / n
    else:
        sc_norm = (succ_count - sc_min) / (sc_max - sc_min)
    
    return alpha * cr_norm + beta * dw_norm + gamma * sc_norm

def priority(instance):
    """
    Computes priority scores based on the Critical Ratio heuristic with composite blending.
    Score = weighted combination of critical ratio, downstream workload, and successor count.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # 15s buffer before 150s limit
    
    # 1. Compute downstream workload, successors, and successor counts
    downstream_workload, successors, succ_count = _compute_downstream_workload(times, predecessors, n)
    
    # 2. Compute Critical Ratio
    # Avoid division by zero
    safe_times = np.where(times == 0, 1e-9, times)
    critical_ratio = downstream_workload / safe_times
    
    # 3. Compute composite priority with different weight combinations
    rng = np.random.default_rng(42)
    
    # Try several weight combinations to find the best base
    weight_combos = [
        (0.5, 0.3, 0.2),  # Balanced
        (0.6, 0.2, 0.2),  # Critical ratio heavy
        (0.4, 0.4, 0.2),  # Downstream workload heavy
        (0.5, 0.2, 0.3),  # Successor count heavy
        (0.3, 0.3, 0.4),  # Equal-ish with more succ count
    ]
    
    best_priority_scores = None
    best_stations = float('inf')
    
    total_time = np.sum(times)
    lower_bound = int(np.ceil(total_time / cycle_time))
    
    for alpha, beta, gamma in weight_combos:
        if time.time() > deadline - 30.0:
            break
        
        candidate = _composite_priority(critical_ratio, downstream_workload, succ_count, times, alpha, beta, gamma)
        # Add small deterministic tie-breaker based on task index to avoid exact ties
        candidate = candidate + np.arange(n) * 1e-9
        
        stations = _simulate_greedy(candidate, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate.copy()
            
            if best_stations == lower_bound:
                break
    
    # If no valid base found, fall back to pure critical ratio
    if best_priority_scores is None:
        best_priority_scores = critical_ratio.copy() + np.arange(n) * 1e-9
        best_stations = _simulate_greedy(best_priority_scores, times, predecessors, cycle_time)
    
    # 4. Local Search: Noise perturbation
    noise_scales = [0.01, 0.05, 0.1, 0.2]
    
    for scale in noise_scales:
        if time.time() > deadline - 20.0:
            break
        
        noise = rng.uniform(-scale, scale, n)
        candidate_scores = best_priority_scores + noise
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            if best_stations == lower_bound:
                break
                
    # 5. Random restarts with different noise seeds
    iterations = 0
    max_iterations = 80
    while time.time() < deadline - 15.0 and iterations < max_iterations:
        iterations += 1
        # Random perturbation with varying scales
        scale = rng.uniform(0.05, 0.3)
        noise = rng.uniform(-scale, scale, n)
        candidate_scores = best_priority_scores + noise
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            if best_stations == lower_bound:
                break

    # 6. Pairwise swap local search: try swapping priority ranks of close-ranked tasks
    # This explores the discrete ordering space more effectively than random noise.
    # Sort tasks by current priority to identify adjacent/close pairs.
    order = np.argsort(-best_priority_scores)  # descending priority
    n_swaps = 0
    max_swaps = 300
    
    while time.time() < deadline - 5.0 and n_swaps < max_swaps:
        n_swaps += 1
        
        # Pick a random pair of tasks that are close in rank
        # Focus on swaps within a small rank window to maintain structural integrity
        i_idx = int(rng.integers(0, n))
        # Choose j within a small rank distance (1-3 ranks away)
        rank_i = int(np.where(order == i_idx)[0][0])
        rank_offset = int(rng.integers(1, 4))
        if rank_i - rank_offset >= 0:
            rank_j = rank_i - rank_offset
        elif rank_i + rank_offset < n:
            rank_j = rank_i + rank_offset
        else:
            continue
        
        j_idx = int(order[rank_j])
        
        if i_idx == j_idx:
            continue
        
        # Try swapping the priorities of these two tasks
        candidate_scores = best_priority_scores.copy()
        temp = candidate_scores[i_idx]
        candidate_scores[i_idx] = candidate_scores[j_idx]
        candidate_scores[j_idx] = temp
        
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            # Update order after improvement
            order = np.argsort(-best_priority_scores)
            
            if best_stations == lower_bound:
                break
    
    # 7. Final polish: try more aggressive swaps if time remains
    if time.time() < deadline - 2.0 and best_stations > lower_bound:
        order = np.argsort(-best_priority_scores)
        n_polish = 0
        max_polish = 100
        
        while time.time() < deadline - 1.0 and n_polish < max_polish:
            n_polish += 1
            
            # Try swapping tasks that are farther apart in rank
            i_idx = int(rng.integers(0, n))
            rank_i = int(np.where(order == i_idx)[0][0])
            # Pick a task 3-10 ranks away
            max_offset = min(10, n - 1)
            rank_offset = int(rng.integers(3, max_offset + 1))
            if rank_i - rank_offset >= 0:
                rank_j = rank_i - rank_offset
            elif rank_i + rank_offset < n:
                rank_j = rank_i + rank_offset
            else:
                continue
            
            j_idx = int(order[rank_j])
            
            if i_idx == j_idx:
                continue
            
            candidate_scores = best_priority_scores.copy()
            temp = candidate_scores[i_idx]
            candidate_scores[i_idx] = candidate_scores[j_idx]
            candidate_scores[j_idx] = temp
            
            stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time)
            
            if stations < best_stations:
                best_stations = stations
                best_priority_scores = candidate_scores.copy()
                order = np.argsort(-best_priority_scores)
                
                if best_stations == lower_bound:
                    break
    
    return best_priority_scores
