"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _topo_order_and_successors(n, preds):
    """Compute a topological order and the list of successors for each node.
    
    Returns:
        topo: list of nodes in topological order (index i is the (i+1)-th in topo)
        succs: list of lists, succs[i] = list of successors of node i
        pos_in_topo: array where pos_in_topo[i] = position of node i in topo order
    """
    # Build successor lists
    succs = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if p < n:  # safety
                succs[p].append(i)
                in_degree[i] += 1
    
    # Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo = []
    pos_in_topo = np.zeros(n, dtype=int)
    while queue:
        u = queue.popleft()
        topo.append(u)
        pos_in_topo[u] = len(topo) - 1
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If there's a cycle (shouldn't happen in valid instances), append remaining
    if len(topo) < n:
        for i in range(n):
            if pos_in_topo[i] == 0 and i not in set(topo):
                topo.append(i)
                pos_in_topo[i] = len(topo) - 1
        # Handle any remaining
        in_topo = set(topo)
        for i in range(n):
            if i not in in_topo:
                topo.append(i)
                pos_in_topo[i] = len(topo) - 1
    
    return topo, succs, pos_in_topo


def _longest_path_to_end(n, succs, times, pos_in_topo, topo):
    """Compute longest path from each node to the end (in reverse topological order)."""
    # Process in reverse topological order
    lp = np.zeros(n, dtype=float)
    # topo is in forward order, so reverse it
    for idx in range(len(topo) - 1, -1, -1):
        u = topo[idx]
        t = times[u]
        max_succ = 0.0
        for v in succs[u]:
            if lp[v] > max_succ:
                max_succ = lp[v]
        lp[u] = t + max_succ
    return lp


def _successor_count(n, succs):
    """Count immediate successors for each node."""
    sc = np.zeros(n, dtype=float)
    for i in range(n):
        sc[i] = len(succs[i])
    return sc


def _simulate_station_count(n, times, preds, cycle_time, priority_scores):
    """Simulate the greedy station-filling algorithm and return number of stations.
    
    Stations are filled with the highest-priority eligible task that fits.
    Eligible = all predecessors already assigned to previous stations.
    """
    # Convert preds to a set for quick lookup
    pred_sets = [set(p for p in preds[i] if p < n) for i in range(n)]
    
    assigned = [False] * n
    stations = 0
    remaining = list(range(n))
    
    while remaining:
        stations += 1
        # Find eligible tasks: all predecessors are assigned
        eligible = []
        for i in remaining:
            if all(assigned[p] for p in pred_sets[i]):
                if times[i] <= cycle_time:
                    eligible.append(i)
        
        if not eligible:
            # Deadlock - shouldn't happen, but break to avoid infinite loop
            break
        
        # Sort eligible by priority (descending)
        eligible.sort(key=lambda x: priority_scores[x], reverse=True)
        
        # Fill current station greedily
        station_load = 0.0
        for i in eligible:
            if station_load + times[i] <= cycle_time:
                assigned[i] = True
                station_load += times[i]
                remaining.remove(i)
    
    return stations


def priority(instance):
    """Score each assembly task for the greedy station-filling heuristic.
    
    Uses a multi-criteria priority combining:
    - Longest path to end (critical path length)
    - Number of successors (breadth of dependency)
    - Topological position (earlier = higher priority)
    - Task time (longer = higher priority)
    """
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute structural features
    topo, succs, pos_in_topo = _topo_order_and_successors(n, preds)
    lp = _longest_path_to_end(n, succs, times, pos_in_topo, topo)
    sc = _successor_count(n, succs)
    
    # Normalize features to [0, 1] range
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    lp_norm = normalize(lp)
    sc_norm = normalize(sc)
    time_norm = normalize(times)
    
    # Invert topological position: earlier in topo = higher priority
    pos_norm = 1.0 - normalize(pos_in_topo.astype(float))
    
    # Multi-criteria priority: weighted sum
    # Weights tuned to emphasize critical path and task time
    w_lp = 0.40
    w_time = 0.30
    w_sc = 0.15
    w_pos = 0.15
    
    scores = w_lp * lp_norm + w_time * time_norm + w_sc * sc_norm + w_pos * pos_norm
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    # Try to refine weights by local search if time permits
    # We'll do a simple coordinate descent on weights
    best_score = scores.copy()
    best_stations = _simulate_station_count(n, times, preds, cycle_time, best_score)
    
    # Try a few perturbations of the weights
    base_weights = np.array([w_lp, w_time, w_sc, w_pos])
    base_norms = [lp_norm, time_norm, sc_norm, pos_norm]
    
    # Coordinate descent: try adjusting each weight
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for i in range(4):
            for delta in [-0.05, 0.05, -0.1, 0.1]:
                w = base_weights.copy()
                w[i] += delta
                if w[i] < 0:
                    w[i] = 0
                # Renormalize to sum to 1
                w_sum = w.sum()
                if w_sum < 1e-12:
                    continue
                w = w / w_sum
                trial_scores = np.zeros(n)
                for j in range(4):
                    trial_scores += w[j] * base_norms[j]
                trial_stations = _simulate_station_count(n, times, preds, cycle_time, trial_scores)
                if trial_stations < best_stations:
                    best_stations = trial_stations
                    best_score = trial_scores
                    base_weights = w
                    improved = True
                    if time.time() > deadline - 10:
                        break
            if time.time() > deadline - 10:
                break
    
    return best_score
# EVOLVE-BLOCK-END
