"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _simulate(priority_scores, times, predecessors, successors, n, cycle_time):
    """
    Simulate greedy station filling with given priority scores.
    Returns number of stations used.
    """
    completed = np.zeros(n, dtype=bool)
    stations = 0
    current_load = 0.0
    
    # Precompute predecessor sets for fast lookup
    pred_sets = [set(predecessors[i]) if i < len(predecessors) else set() for i in range(n)]
    
    while not np.all(completed):
        # Find eligible tasks: all preds completed, fits in remaining cycle time
        eligible = []
        for task in range(n):
            if completed[task]:
                continue
            if all(completed[p] for p in pred_sets[task]):
                if times[task] <= cycle_time - current_load:
                    eligible.append(task)
        
        if not eligible and current_load > 0:
            # Move to next station
            stations += 1
            current_load = 0.0
            for task in range(n):
                if completed[task]:
                    continue
                if all(completed[p] for p in pred_sets[task]):
                    if times[task] <= cycle_time:
                        eligible.append(task)
        
        if not eligible:
            # Infeasible or bug; break to avoid infinite loop
            stations += 1
            break
        
        # Pick highest priority
        best_task = max(eligible, key=lambda t: priority_scores[t])
        completed[best_task] = True
        current_load += times[best_task]
    
    return stations

def _compute_features(times, predecessors, successors, n):
    """Compute various priority features for tasks."""
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        preds = predecessors[j] if j < len(predecessors) else []
        in_degree[j] = len(preds)
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining)
    
    # Positional weight (longest path to end)
    pos_weights = times.copy()
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
    
    # Depth from start (longest path from any source to i)
    depth_from_start = np.zeros(n)
    for u in topo_order:
        for v in successors[u]:
            depth_from_start[v] = max(depth_from_start[v], depth_from_start[u] + times[u])
    
    # Immediate counts
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    ipc = np.array([len(predecessors[i]) if i < len(predecessors) else 0 for i in range(n)], dtype=float)
    
    # Normalize features to [0, 1]
    max_pw = np.max(pos_weights) if np.max(pos_weights) > 0 else 1
    max_t = np.max(times) if np.max(times) > 0 else 1
    max_dfs = np.max(depth_from_start) if np.max(depth_from_start) > 0 else 1
    max_isc = np.max(isc) if np.max(isc) > 0 else 1
    max_ipc = np.max(ipc) if np.max(ipc) > 0 else 1
    
    features = {
        "pos_weights": pos_weights / max_pw,
        "times": times / max_t,
        "isc": isc / max_isc,
        "ipc": ipc / max_ipc,
        "depth_from_start": depth_from_start / max_dfs,
        "raw_pos_weights": pos_weights,
        "raw_times": times,
        "raw_isc": isc,
        "raw_ipc": ipc,
        "raw_depth": depth_from_start
    }
    
    return features, topo_order

def _local_search_weights(features, times, predecessors, successors, n, cycle_time, 
                          start_weights=None, max_iter=50, deadline=None):
    """
    Local search over weight combinations to find priority vector that minimizes stations.
    """
    feature_keys = ["pos_weights", "times", "isc", "ipc", "depth_from_start"]
    num_features = len(feature_keys)
    
    # Initialize weights
    if start_weights is None:
        weights = np.array([0.55, 0.30, 0.10, -0.05, 0.10])
    else:
        weights = start_weights.copy()
    
    # Compute initial priority
    feature_matrix = np.column_stack([features[k] for k in feature_keys])
    priority = feature_matrix @ weights
    
    # Normalize to avoid numerical issues
    if np.max(priority) - np.min(priority) > 1e-6:
        priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    
    best_stations = _simulate(priority, times, predecessors, successors, n, cycle_time)
    best_weights = weights.copy()
    
    # Local search: perturb weights and keep if improvement
    step_size = 0.05
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        for dim in range(num_features):
            # Try increasing weight
            trial_weights = best_weights.copy()
            trial_weights[dim] += step_size
            trial_priority = feature_matrix @ trial_weights
            if np.max(trial_priority) - np.min(trial_priority) > 1e-6:
                trial_priority = (trial_priority - np.min(trial_priority)) / (np.max(trial_priority) - np.min(trial_priority))
            trial_stations = _simulate(trial_priority, times, predecessors, successors, n, cycle_time)
            if trial_stations < best_stations:
                best_stations = trial_stations
                best_weights = trial_weights
                improved = True
                break
        
        # Try decreasing weight
        for dim in range(num_features):
            trial_weights = best_weights.copy()
            trial_weights[dim] -= step_size
            trial_priority = feature_matrix @ trial_weights
            if np.max(trial_priority) - np.min(trial_priority) > 1e-6:
                trial_priority = (trial_priority - np.min(trial_priority)) / (np.max(trial_priority) - np.min(trial_priority))
            trial_stations = _simulate(trial_priority, times, predecessors, successors, n, cycle_time)
            if trial_stations < best_stations:
                best_stations = trial_stations
                best_weights = trial_weights
                improved = True
                break
        
        if not improved:
            # Try smaller step
            step_size *= 0.7
            if step_size < 0.01:
                break
    
    # Compute final priority from best weights
    final_priority = feature_matrix @ best_weights
    if np.max(final_priority) - np.min(final_priority) > 1e-6:
        final_priority = (final_priority - np.min(final_priority)) / (np.max(final_priority) - np.min(final_priority))
    
    return final_priority, best_stations

def _slack_based_priority(instance, times, predecessors, successors, n, cycle_time):
    """
    Compute priority based on slack: negative of the difference between 
    processing time and remaining cycle time. This forces packing of tightest tasks first.
    """
    # For each task, compute a slack-based priority
    # Slack = cycle_time - times[i]
    # We want to prioritize tasks with LESS slack (tighter fit)
    # So priority = -slack = times[i] - cycle_time (but this is always negative)
    # Better: priority = times[i] / cycle_time (normalized tightness)
    
    # Combine with positional weight for better results
    features, topo_order = _compute_features(times, predecessors, successors, n)
    
    # Tightness: how close the task is to filling the cycle time
    tightness = times / cycle_time
    
    # Normalize tightness to [0, 1]
    max_tight = np.max(tightness) if np.max(tightness) > 0 else 1
    tightness_norm = tightness / max_tight
    
    # Combine with positional weight and other features
    # Weighting: emphasize tightness heavily, then positional weight
    w_tight = 0.40
    w_pw = 0.35
    w_isc = 0.15
    w_time = 0.10
    
    priority = (w_tight * tightness_norm + 
                w_pw * features["pos_weights"] + 
                w_isc * features["isc"] +
                w_time * features["times"])
    
    # Normalize to [0, 1]
    if np.max(priority) - np.min(priority) > 1e-6:
        priority = (priority - np.min(priority)) / (np.max(priority) - np.min(priority))
    
    return priority

def priority(instance):
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    cycle_time = instance.get("cycle_time", 40)
    
    # Build successor graph
    successors = [[] for _ in range(n)]
    for j in range(n):
        preds = predecessors[j] if j < len(predecessors) else []
        for i in preds:
            successors[i].append(j)
    
    # Compute features
    features, topo_order = _compute_features(times, predecessors, successors, n)
    
    # Try slack-based priority first
    slack_priority = _slack_based_priority(instance, times, predecessors, successors, n, cycle_time)
    slack_stations = _simulate(slack_priority, times, predecessors, successors, n, cycle_time)
    
    # Check time
    if time.time() - start_time > deadline - 10:
        return slack_priority
    
    # Try local search to find better weight combination
    try:
        # Use normalized features for local search
        feature_keys = ["pos_weights", "times", "isc", "ipc", "depth_from_start"]
        feature_matrix = np.column_stack([features[k] for k in feature_keys])
        
        # Initial weights: emphasize tightness (times) and positional weight
        init_weights = np.array([0.35, 0.40, 0.15, -0.05, 0.10])
        
        # Run local search
        search_priority, search_stations = _local_search_weights(
            features, times, predecessors, successors, n, cycle_time,
            start_weights=init_weights,
            max_iter=40,
            deadline=deadline - 5
        )
        
        # Check if search improved
        if search_stations < slack_stations:
            return search_priority
        else:
            return slack_priority
            
    except Exception:
        return slack_priority
# EVOLVE-BLOCK-END
