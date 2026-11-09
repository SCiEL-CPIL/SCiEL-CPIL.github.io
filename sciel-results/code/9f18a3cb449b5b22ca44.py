"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

# Global timer to ensure we don't exceed the 120s budget across all instances
_start_time = None
_TIME_LIMIT = 115.0  # Safety margin

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _remaining_time():
    return _TIME_LIMIT - _get_elapsed()

def _compute_successors(predecessors, n):
    """Compute successors list from predecessors."""
    succ = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            succ[i].append(j)
    return succ

def _compute_longest_path_downstream(times, succ, n):
    """
    Compute longest path from each task to the end (downstream).
    """
    longest = np.array(times, dtype=float).copy()
    # Topological sort order is needed for exact calculation, 
    # but for N=30, iterative relaxation is fast enough.
    for _ in range(n):
        changed = False
        for i in range(n):
            if succ[i]:
                max_down = max(longest[j] for j in succ[i])
                new_val = times[i] + max_down
                if new_val > longest[i] + 1e-9:
                    longest[i] = new_val
                    changed = True
        if not changed:
            break
    return longest

def _compute_longest_path_upstream(times, predecessors, n):
    """
    Compute longest path from the start to each task (upstream).
    """
    longest_up = np.array(times, dtype=float).copy()
    for _ in range(n):
        changed = False
        for i in range(n):
            if predecessors[i]:
                max_up = max(longest_up[j] for j in predecessors[i])
                new_val = times[i] + max_up
                if new_val > longest_up[i] + 1e-9:
                    longest_up[i] = new_val
                    changed = True
        if not changed:
            break
    return longest_up

def _simulate(priority_scores, times, predecessors, cycle_time, n, succ):
    """
    Simulate the greedy station-filling algorithm.
    Returns the number of stations used.
    """
    completed = [False] * n
    remaining_pred = [len(predecessors[j]) for j in range(n)]
    
    stations = 0
    current_station_time = 0.0
    in_station = False
    
    while any(not c for c in completed):
        # Find eligible tasks that fit in the current station
        eligible = []
        for j in range(n):
            if not completed[j] and remaining_pred[j] == 0:
                if current_station_time + times[j] <= cycle_time + 1e-9:
                    eligible.append(j)
        
        if eligible:
            # Pick highest priority
            best_task = max(eligible, key=lambda j: priority_scores[j])
            current_station_time += times[best_task]
            completed[best_task] = True
            in_station = True
            
            for k in succ[best_task]:
                remaining_pred[k] -= 1
        else:
            # No eligible task fits in current station
            if in_station:
                stations += 1
                current_station_time = 0.0
                in_station = False
            else:
                # Start first station
                stations += 1
                current_station_time = 0.0
                in_station = False
                # Re-check: now with empty station
                eligible = []
                for j in range(n):
                    if not completed[j] and remaining_pred[j] == 0:
                        if times[j] <= cycle_time + 1e-9:
                            eligible.append(j)
                
                if not eligible:
                    # Should not happen if cycle_time is valid
                    break
                
                best_task = max(eligible, key=lambda j: priority_scores[j])
                current_station_time += times[best_task]
                completed[best_task] = True
                in_station = True
                
                for k in succ[best_task]:
                    remaining_pred[k] -= 1
    
    if in_station:
        stations += 1
    
    return stations

def priority(instance):
    """
    Score each assembly task to minimize the number of stations.
    Uses a local search to find optimal priority weights for the specific instance.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # Precompute features
    succ = _compute_successors(predecessors, n)
    longest_down = _compute_longest_path_downstream(times, succ, n)
    longest_up = _compute_longest_path_upstream(times, predecessors, n)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Normalize features to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-12:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    feat_down = normalize(longest_down)
    feat_up = normalize(longest_up)
    feat_succ = normalize(num_succ)
    feat_pred = normalize(num_pred)
    feat_time = normalize(times)
    
    # Feature matrix: rows are tasks, cols are features
    # Features: Down, Up, Succ, Pred, Time
    features = np.column_stack([feat_down, feat_up, feat_succ, feat_pred, feat_time])
    
    # Search for best weights
    # Weights are non-negative. We can scale them arbitrarily, so we fix one weight to 1 or normalize weights.
    # Let's use a simplex-like search or just random vectors.
    
    best_stations = float('inf')
    best_weights = np.ones(5) # Default: equal weights
    
    # Get a baseline first
    baseline_scores = features @ best_weights
    baseline_stations = _simulate(baseline_scores, times, predecessors, cycle_time, n, succ)
    if baseline_stations < best_stations:
        best_stations = baseline_stations
        best_weights = best_weights.copy()
        
    # Random Restart Hill Climbing
    # Time budget check
    max_time = _remaining_time()
    # We want to leave some time for other instances if this is part of a batch, 
    # but we don't know the total count. The contract says "five generated instances".
    # So we have 120s / 5 = 24s per instance on average.
    # Let's allow up to 10s per instance to be safe.
    time_budget_per_instance = 10.0
    if max_time < time_budget_per_instance:
        # If we are running out of total time, use what's left
        time_budget_per_instance = max(0.1, max_time)
        
    start_search_time = time.time()
    rng = np.random.default_rng(42)
    
    num_restarts = 20
    steps_per_restart = 500
    
    for r in range(num_restarts):
        if time.time() - start_search_time > time_budget_per_instance:
            break
            
        # Initialize random weights
        w = rng.random(5)
        # Ensure non-negative
        w = np.abs(w)
        if np.sum(w) < 1e-6:
            w = np.ones(5)
            
        # Hill climbing
        current_stations = _simulate(features @ w, times, predecessors, cycle_time, n, succ)
        
        for step in range(steps_per_restart):
            if time.time() - start_search_time > time_budget_per_instance:
                break
                
            # Perturb a random subset of weights
            num_perturb = rng.integers(1, 3)
            indices = rng.choice(5, size=num_perturb, replace=False)
            
            new_w = w.copy()
            for idx in indices:
                new_w[idx] += rng.normal(0, 0.5)
                new_w[idx] = max(0, new_w[idx])
            
            # Normalize weights
            if np.sum(new_w) > 1e-6:
                new_w = new_w / np.sum(new_w)
            else:
                new_w = np.ones(5) / 5.0
                
            new_stations = _simulate(features @ new_w, times, predecessors, cycle_time, n, succ)
            
            if new_stations < current_stations:
                w = new_w
                current_stations = new_stations
            elif new_stations == current_stations and rng.random() < 0.1:
                # Accept equal with small probability to escape local minima
                w = new_w
                current_stations = new_stations
                
        if current_stations < best_stations:
            best_stations = current_stations
            best_weights = w.copy()
            
    # Final polish: try small perturbations around best_weights
    if _remaining_time() > 1.0:
        polish_start = time.time()
        for _ in range(200):
            if time.time() - polish_start > 1.0:
                break
            w = best_weights.copy()
            idx = rng.integers(0, 5)
            w[idx] += rng.normal(0, 0.1)
            w[idx] = max(0, w[idx])
            if np.sum(w) > 1e-6:
                w = w / np.sum(w)
            else:
                w = np.ones(5) / 5.0
            new_stations = _simulate(features @ w, times, predecessors, cycle_time, n, succ)
            if new_stations < best_stations:
                best_stations = new_stations
                best_weights = w.copy()
    
    # Return the best priority found
    result = features @ best_weights
    
    return result

# EVOLVE-BLOCK-END
