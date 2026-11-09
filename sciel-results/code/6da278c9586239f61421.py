"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _simulate_station_count(instance, scores):
    """Simulate the greedy station filling and return number of stations used."""
    times = instance["times"]
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    
    # Compute in-degrees
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
    
    # Priority queue: (score, task_id) - we want highest score first
    # Use a list and sort each time (n is small ~30)
    eligible = []
    for i in range(n):
        if in_degree[i] == 0:
            eligible.append(i)
    
    stations = 0
    remaining = np.ones(n, dtype=bool)
    current_station_load = 0
    
    while remaining.sum() > 0:
        # Find eligible tasks (all predecessors completed)
        eligible = [i for i in range(n) if remaining[i] and in_degree[i] == 0]
        
        if not eligible:
            break
        
        # Sort by priority (descending)
        eligible.sort(key=lambda x: scores[x], reverse=True)
        
        # Try to fit tasks into current station
        placed_any = False
        for i in eligible:
            if times[i] <= cycle_time - current_station_load:
                # Place task i
                remaining[i] = False
                current_station_load += times[i]
                placed_any = True
                # Update in-degrees of successors
                for s in successors[i]:
                    in_degree[s] -= 1
                break  # Only place one task per iteration to maintain greedy order
        
        if not placed_any:
            # Start a new station
            stations += 1
            current_station_load = 0
    
    return stations + 1  # +1 for the last station if any tasks remain


def _compute_ranks(instance):
    """Compute various rank-based features for tasks."""
    times = instance["times"]
    preds = instance["predecessors"]
    n = len(times)
    
    # Successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    
    # Longest path from task i to any terminal (including i's own time)
    longest_path = np.zeros(n, dtype=float)
    # Topological order
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Compute longest path in reverse topological order
    for i in reversed(topo_order):
        max_succ = 0
        for s in successors[i]:
            max_succ = max(max_succ, longest_path[s])
        longest_path[i] = times[i] + max_succ
    
    # Number of successors (direct)
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Depth from start (longest path from any source to i)
    depth_from_start = np.zeros(n, dtype=float)
    for i in topo_order:
        max_pred = 0
        for p in preds[i]:
            max_pred = max(max_pred, depth_from_start[p])
        depth_from_start[i] = max_pred + times[i]
    
    return {
        'longest_path': longest_path,
        'num_successors': num_successors,
        'depth_from_start': depth_from_start,
        'time': times.copy()
    }


def priority(instance):
    """
    Score each assembly task using a slack-based dynamic priority with local search.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    
    # Initial priority: longest task first (baseline)
    best_scores = times.copy()
    best_stations = _simulate_station_count(instance, best_scores)
    
    if time.time() > deadline - 10:
        return best_scores
    
    # Get feature ranks
    ranks = _compute_ranks(instance)
    
    # Define candidate weight vectors to combine features
    # We'll do a simple hill-climbing search over weight combinations
    features = [
        ranks['time'],
        ranks['longest_path'],
        ranks['num_successors'],
        ranks['depth_from_start']
    ]
    
    # Normalize features to [0, 1] range for consistent scaling
    norm_features = []
    for f in features:
        f_min = f.min()
        f_max = f.max()
        if f_max > f_min:
            norm_features.append((f - f_min) / (f_max - f_min))
        else:
            norm_features.append(np.ones(n) * 0.5)
    
    # Start with uniform weights
    weights = np.ones(len(features)) / len(features)
    scores = sum(w * f for w, f in zip(weights, norm_features))
    
    # Local search: hill climbing on weight vector
    random.seed(42)
    max_iterations = 200
    step_size = 0.2
    
    current_stations = _simulate_station_count(instance, scores)
    if current_stations < best_stations:
        best_stations = current_stations
        best_scores = scores.copy()
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Perturb weights
        new_weights = weights.copy()
        idx = random.randint(0, len(features) - 1)
        delta = random.uniform(-step_size, step_size)
        new_weights[idx] = max(0, new_weights[idx] + delta)
        
        # Normalize weights to sum to 1
        w_sum = new_weights.sum()
        if w_sum > 0:
            new_weights = new_weights / w_sum
        else:
            new_weights = np.ones(len(features)) / len(features)
        
        # Compute new scores
        new_scores = sum(w * f for w, f in zip(new_weights, norm_features))
        
        # Evaluate
        new_stations = _simulate_station_count(instance, new_scores)
        
        if new_stations < best_stations:
            best_stations = new_stations
            best_scores = new_scores.copy()
            weights = new_weights.copy()
            scores = new_scores.copy()
        elif new_stations < current_stations:
            # Accept if better than current (hill climbing)
            weights = new_weights.copy()
            scores = new_scores.copy()
            current_stations = new_stations
        else:
            # Occasionally accept worse solutions for diversity
            if random.random() < 0.1:
                weights = new_weights.copy()
                scores = new_scores.copy()
                current_stations = new_stations
    
    return best_scores
# EVOLVE-BLOCK-END
