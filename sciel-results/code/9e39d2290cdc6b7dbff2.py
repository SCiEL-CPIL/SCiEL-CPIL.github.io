import numpy as np
from collections import defaultdict, deque
import time

def _build_graph(predecessors, n):
    """Build successor list and in-degree array."""
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        if isinstance(predecessors, dict):
            preds = predecessors.get(i, [])
        else:
            preds = predecessors[i]
        for p in preds:
            succ[p].append(i)
            in_degree[i] += 1
    return succ, in_degree

def _topo_sort(succ, in_degree, n):
    """Kahn's algorithm. Returns (topo_order, temp_in_degree)."""
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    return topo_order

def _compute_features(times, succ, in_degree, topo_order, n):
    """Compute rank, down_work, out_degree."""
    rank = [0.0] * n
    down_work = [0.0] * n
    for u in reversed(topo_order):
        max_succ_rank = 0.0
        sum_succ = 0.0
        for v in succ[u]:
            if rank[v] > max_succ_rank:
                max_succ_rank = rank[v]
            sum_succ += times[v] + down_work[v]
        rank[u] = times[u] + max_succ_rank
        down_work[u] = sum_succ
    
    out_degree = np.array([len(succ[i]) for i in range(n)], dtype=float)
    in_deg = np.array(in_degree, dtype=float)
    
    return np.array(rank), np.array(down_work), out_degree, in_deg

def _normalize(arr):
    """Normalize array to [0, 1]."""
    arr = np.asarray(arr, dtype=float)
    mn = arr.min()
    mx = arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _simulate_greedy(priority_scores, times, predecessors, cycle_time, n, succ, in_degree):
    """
    Simulates the greedy assignment process.
    Returns the number of stations used.
    """
    assigned = [False] * n
    station_load = 0.0
    num_stations = 1
    remaining_count = n
    # Use a list for remaining tasks, remove by index swap for O(1) removal
    remaining = list(range(n))
    remaining_indices = {i: idx for idx, i in enumerate(remaining)}
    
    while remaining_count > 0:
        best_task_idx = -1
        best_priority = -np.inf
        
        # Find the highest priority task that is eligible and fits
        for i in remaining:
            # Check eligibility: all predecessors must be assigned
            eligible = True
            if isinstance(predecessors, dict):
                preds = predecessors.get(i, [])
            else:
                preds = predecessors[i]
            
            for p in preds:
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
            remaining_count -= 1
            # Remove from remaining efficiently
            idx = remaining_indices[best_task_idx]
            last_idx = remaining_count
            if idx != last_idx:
                remaining[idx] = remaining[last_idx]
                remaining_indices[remaining[idx]] = idx
            del remaining_indices[best_task_idx]
            remaining = remaining[:last_idx]
        else:
            # No task fits in the current station. Start a new station.
            num_stations += 1
            station_load = 0.0
            
            # In a new station, find the highest priority eligible task that fits (time <= cycle_time)
            best_task_idx = -1
            best_priority = -np.inf
            
            for i in remaining:
                if assigned[i]:
                    continue
                
                # Check eligibility
                if isinstance(predecessors, dict):
                    preds = predecessors.get(i, [])
                else:
                    preds = predecessors[i]
                
                eligible = True
                for p in preds:
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
                remaining_count -= 1
                idx = remaining_indices[best_task_idx]
                last_idx = remaining_count
                if idx != last_idx:
                    remaining[idx] = remaining[last_idx]
                    remaining_indices[remaining[idx]] = idx
                del remaining_indices[best_task_idx]
                remaining = remaining[:last_idx]
            else:
                # Should not happen in feasible instances, but break to avoid infinite loop
                break
                
    return num_stations

def priority(instance):
    """
    Compute priority scores for assembly tasks.
    Higher priority = more important to assign earlier.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance.get("cycle_time", None)
    
    n = len(times)
    if n == 0:
        return np.array([], dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer before 150s limit
    
    # Build graph structures
    succ, in_degree = _build_graph(predecessors, n)
    
    # Topological sort
    topo_order = _topo_sort(succ, in_degree, n)
    
    if len(topo_order) < n:
        # Cycle detected - fallback to times
        return times
    
    # Compute features
    rank, down_work, out_degree, in_deg = _compute_features(times, succ, in_degree, topo_order, n)
    
    # Normalize each component to [0, 1] range
    norm_rank = _normalize(rank)
    norm_down = _normalize(down_work)
    norm_outdeg = _normalize(out_degree)
    norm_time = _normalize(times)
    
    # Base score from current program's weighted combination
    # Emphasize critical path position and downstream impact
    base_score = (1.2 * norm_rank + 
                  0.8 * norm_down + 
                  0.5 * norm_outdeg + 
                  0.3 * norm_time)
    
    # Add small tie-breaking by task duration to ensure deterministic ordering
    eps = 1e-9
    base_score = base_score + eps * norm_time
    
    # Compute Critical Ratio from inspiration
    # Ratio = (Sum of times of all successors + own time) / own time
    # This is (down_work[u] + times[u]) / times[u]
    safe_times = np.where(times == 0, 1e-9, times)
    critical_ratio = (down_work + times) / safe_times
    norm_crit_ratio = _normalize(critical_ratio)
    
    # Combine base score and critical ratio
    # We'll use a weighted combination, but also do local search to find the best weight
    # Let alpha be the weight of critical ratio, and (1-alpha) be the weight of base_score
    # We'll search over alpha in [0, 1]
    
    best_priority_scores = base_score.copy()
    best_stations = _simulate_greedy(best_priority_scores, times, predecessors, cycle_time, n, succ, in_degree)
    
    total_time = np.sum(times)
    lower_bound = int(np.ceil(total_time / cycle_time))
    
    # Search for optimal alpha
    # We'll try a grid of alpha values
    alphas = np.linspace(0, 1, 21)  # 21 values from 0 to 1
    
    for alpha in alphas:
        if time.time() > deadline - 5.0:
            break
        
        candidate_scores = (1 - alpha) * base_score + alpha * norm_crit_ratio
        stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time, n, succ, in_degree)
        
        if stations < best_stations:
            best_stations = stations
            best_priority_scores = candidate_scores.copy()
            
            # Check lower bound
            if best_stations == lower_bound:
                break
    
    # If time remains, do a finer search around the best alpha found
    # We can also try adding small noise to break ties
    if time.time() < deadline - 2.0:
        rng = np.random.default_rng(42)
        
        # Try a few random perturbations
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            
            # Add small noise
            noise = rng.uniform(-0.01, 0.01, n)
            candidate_scores = best_priority_scores + noise
            
            stations = _simulate_greedy(candidate_scores, times, predecessors, cycle_time, n, succ, in_degree)
            
            if stations < best_stations:
                best_stations = stations
                best_priority_scores = candidate_scores.copy()
                
                if best_stations == lower_bound:
                    break
    
    return best_priority_scores
