"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_longest_paths(instance):
    """Compute longest path from each node to any sink (including own time)."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", None)
    n = len(times)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    if preds is not None:
        # preds[i] is a list of predecessors of task i
        for i in range(n):
            if preds[i] is not None:
                for p in preds[i]:
                    successors[p].append(i)
                    in_degree[i] += 1
    
    # Topological order (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Longest path from node to sink (reverse topological order)
    longest_to_sink = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        longest_to_sink[u] = times[u]
        for v in successors[u]:
            if longest_to_sink[u] < times[u] + longest_to_sink[v]:
                longest_to_sink[u] = times[u] + longest_to_sink[v]
    
    return longest_to_sink, successors, topo_order


def _simulate_station_count(instance, priorities):
    """Simulate station filling and return number of stations."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    preds = instance.get("predecessors", None)
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                for p in preds[i]:
                    successors[p].append(i)
                    in_degree[i] += 1
    
    # Sort tasks by priority (descending)
    order = np.argsort(-priorities)
    
    assigned = [False] * n
    station_count = 0
    current_station_load = 0.0
    # Track which tasks are available (all predecessors assigned)
    avail_count = [in_degree[i] for i in range(n)]
    
    for station in range(1000):  # Safety bound
        station_count = station + 1
        current_station_load = 0.0
        # Try to add tasks to this station in priority order
        for task in order:
            if assigned[task]:
                continue
            if avail_count[task] > 0:
                continue
            if current_station_load + times[task] > cycle_time:
                continue
            # Assign task
            assigned[task] = True
            current_station_load += times[task]
            for succ in successors[task]:
                avail_count[succ] -= 1
        
        if all(assigned):
            break
    
    return station_count


def _compute_depth(instance):
    """Compute depth of each task (longest path from source to task, including own time)."""
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    preds = instance.get("predecessors", None)
    
    # Build predecessor lists
    pred_lists = [[] for _ in range(n)]
    in_degree = [0] * n
    if preds is not None:
        for i in range(n):
            if preds[i] is not None:
                for p in preds[i]:
                    pred_lists[i].append(p)
                    in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    depth = np.zeros(n, dtype=float)
    visited = [False] * n
    while queue:
        u = queue.popleft()
        visited[u] = True
        for i in range(n):
            if u in pred_lists[i]:
                if depth[i] < depth[u] + times[i]:
                    depth[i] = depth[u] + times[i]
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)
    
    return depth


def priority(instance):
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 110s to be safe
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)
    
    # Precompute features
    longest_to_sink, successors, topo_order = _compute_longest_paths(instance)
    depth = _compute_depth(instance)
    
    # Number of successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Base features
    features = np.column_stack([
        times,              # task time
        longest_to_sink,    # longest path to sink
        depth,              # depth from source
        num_successors,     # number of successors
    ])
    
    # Normalize features
    feature_means = features.mean(axis=0)
    feature_stds = features.std(axis=0) + 1e-10
    features_norm = (features - feature_means) / feature_stds
    
    # Start with a good initial priority: weighted combination
    # Try several weight combinations and pick the best
    
    def _eval_weights(weights):
        """Evaluate a weight vector and return station count."""
        priorities = features_norm @ weights
        return _simulate_station_count(instance, priorities)
    
    # Initial weights: emphasize longest path to sink and time
    best_weights = np.array([1.0, 2.0, 0.5, 0.5])
    best_score = _eval_weights(best_weights)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return features_norm @ best_weights
    
    # Local search over weight combinations
    random.seed(42)
    num_features = features_norm.shape[1]
    
    # Try random perturbations of the best weights
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation
        perturbation = np.random.randn(num_features) * 0.3
        new_weights = best_weights + perturbation
        
        new_score = _eval_weights(new_weights)
        if new_score < best_score:
            best_weights = new_weights
            best_score = new_score
    
    # Also try some structured weight combinations
    structured_weights = [
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0, 0.0]),
        np.array([0.5, 2.0, 0.5, 0.5]),
        np.array([1.0, 1.5, 0.5, 0.3]),
        np.array([0.8, 2.0, 0.3, 0.5]),
        np.array([1.0, 1.0, 1.0, 0.0]),
        np.array([0.5, 1.5, 0.5, 0.5]),
    ]
    
    for sw in structured_weights:
        if time.time() > deadline - 5:
            break
        score = _eval_weights(sw)
        if score < best_score:
            best_weights = sw
            best_score = score
    
    # Final local search refinement
    for iteration in range(50):
        if time.time() > deadline - 3:
            break
        perturbation = np.random.randn(num_features) * 0.1
        new_weights = best_weights + perturbation
        new_score = _eval_weights(new_weights)
        if new_score < best_score:
            best_weights = new_weights
            best_score = new_score
    
    return features_norm @ best_weights
# EVOLVE-BLOCK-END
