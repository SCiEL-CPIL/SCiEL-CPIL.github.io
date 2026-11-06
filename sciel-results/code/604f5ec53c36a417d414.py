"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _compute_transitive_successors(predecessors, n):
    """
    Compute the set of all transitive successors for each task.
    Uses BFS/DFS from each node on the reversed graph.
    """
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    trans_succ = [set() for _ in range(n)]
    for i in range(n):
        visited = set()
        queue = deque(successors[i])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for s in successors[node]:
                    if s not in visited:
                        queue.append(s)
        trans_succ[i] = visited
    return trans_succ

def _compute_features(instance):
    """Compute all feature vectors for priority scoring."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    trans_succ = _compute_transitive_successors(predecessors, n)
    trans_succ_counts = np.array([len(s) for s in trans_succ], dtype=float)
    
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    temp_in_degree = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    crit_path = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0
        for s in successors[node]:
            max_succ = max(max_succ, crit_path[s])
        crit_path[node] = times[node] + max_succ
    
    epsilon = 1e-6
    slack = (cycle_time - times + epsilon) / (trans_succ_counts + 1.0)
    
    # Bottleneck slack: ratio of remaining capacity to immediate successor count
    # Tasks with small ratio consume more capacity relative to branching they unlock
    imm_succ_counts = np.array([len(successors[i]) for i in range(n)], dtype=float)
    bottleneck_slack = (cycle_time - times + epsilon) / (imm_succ_counts + 1.0)
    
    max_time = np.max(times) if np.max(times) > 0 else 1
    max_trans = np.max(trans_succ_counts) if np.max(trans_succ_counts) > 0 else 1
    max_crit = np.max(crit_path) if np.max(crit_path) > 0 else 1
    max_slack = np.max(slack) if np.max(slack) > 0 else 1
    max_bneck = np.max(bottleneck_slack) if np.max(bottleneck_slack) > 0 else 1
    
    features = {
        'norm_time': times / max_time,
        'norm_trans': trans_succ_counts / max_trans,
        'norm_crit': crit_path / max_crit,
        'norm_slack': slack / max_slack,
        'norm_bneck': bottleneck_slack / max_bneck,
        'scale': max_time
    }
    
    return features, successors, trans_succ, topo_order

def _greedy_balance(instance, scores):
    """Simulate greedy balancing with given priority scores. Returns number of stations."""
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    remaining = set(range(n))
    placed = [False] * n
    stations = 0
    station_time = 0.0
    
    while remaining:
        # Find eligible tasks (all predecessors placed)
        eligible = []
        for i in remaining:
            all_preds_placed = True
            for pred in predecessors[i]:
                if not placed[pred]:
                    all_preds_placed = False
                    break
            if all_preds_placed and times[i] <= cycle_time - station_time + 1e-9:
                eligible.append(i)
        
        if not eligible:
            # Start new station
            stations += 1
            station_time = 0.0
            eligible = []
            for i in remaining:
                all_preds_placed = True
                for pred in predecessors[i]:
                    if not placed[pred]:
                        all_preds_placed = False
                        break
                if all_preds_placed and times[i] <= cycle_time + 1e-9:
                    eligible.append(i)
        
        if not eligible:
            # No eligible task fits; break to avoid infinite loop
            break
        
        # Pick highest priority eligible task
        best = max(eligible, key=lambda i: scores[i])
        placed[best] = True
        remaining.discard(best)
        station_time += times[best]
    
    return stations

def _compute_ranking(instance):
    """
    Compute priority scores using tuned weights based on greedy simulation feedback.
    Now includes bottleneck slack feature.
    """
    features, successors, trans_succ, topo_order = _compute_features(instance)
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Weights: [time, trans, crit, slack, bneck]
    best_weights = [0.35, 0.25, 0.2, 0.1, 0.1]
    best_stations = float('inf')
    
    # Weight candidates emphasizing bottleneck slack
    weight_candidates = [
        [0.35, 0.25, 0.2, 0.1, 0.1],   # Balanced with bneck
        [0.3, 0.2, 0.2, 0.1, 0.2],     # More bneck
        [0.25, 0.25, 0.2, 0.1, 0.2],   # Even more bneck
        [0.3, 0.25, 0.15, 0.1, 0.2],   # Bneck + trans
        [0.25, 0.2, 0.2, 0.1, 0.25],   # Strong bneck
        [0.3, 0.2, 0.25, 0.05, 0.2],   # Bneck + crit
        [0.2, 0.25, 0.2, 0.1, 0.25],   # Bneck + trans
        [0.35, 0.2, 0.15, 0.1, 0.2],   # Time + bneck
        [0.25, 0.25, 0.2, 0.15, 0.15], # Slack + bneck
        [0.3, 0.3, 0.15, 0.05, 0.2],   # Trans + bneck
        [0.2, 0.2, 0.2, 0.2, 0.2],     # Uniform
        [0.4, 0.2, 0.15, 0.05, 0.2],   # Time-heavy + bneck
    ]
    
    start_time = time.time()
    deadline = start_time + 100.0
    
    for w in weight_candidates:
        if time.time() > deadline:
            break
        scores = (w[0] * features['norm_time'] + 
                  w[1] * features['norm_trans'] + 
                  w[2] * features['norm_crit'] + 
                  w[3] * (1.0 - features['norm_slack']) +
                  w[4] * (1.0 - features['norm_bneck'])) * features['scale']
        scores = np.maximum(scores, 0.1)
        
        try:
            stations = _greedy_balance(instance, scores)
            if stations < best_stations:
                best_stations = stations
                best_weights = w
        except Exception:
            pass
    
    # Use best found weights
    scores = (best_weights[0] * features['norm_time'] + 
              best_weights[1] * features['norm_trans'] + 
              best_weights[2] * features['norm_crit'] + 
              best_weights[3] * (1.0 - features['norm_slack']) +
              best_weights[4] * (1.0 - features['norm_bneck'])) * features['scale']
    scores = np.maximum(scores, 0.1)
    
    return scores

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0
    
    try:
        scores = _compute_ranking(instance)
    except Exception:
        times = np.asarray(instance["times"], dtype=float)
        scores = times
        
    if time.time() > deadline:
        times = np.asarray(instance["times"], dtype=float)
        return times
        
    return scores
# EVOLVE-BLOCK-END
