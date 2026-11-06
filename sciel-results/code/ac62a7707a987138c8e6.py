"""Assembly Line Balancing - Model and Optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque

def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0
    
    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    
    if n > 0 and not eligible:
        return 10**9
    
    station_count = 1
    current_load = 0
    unplaced_count = n
    
    # Pre-allocate a list for sorting to avoid repeated list creation overhead if possible,
    # but for N=30, sorted() is fine.
    
    while eligible:
        # Sort eligible by priority descending
        # Using a list comprehension and sort is faster than sorted with key for small N?
        # Let's stick to sorted.
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
                unplaced_count -= 1
                placed = True
                
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        
        if not placed:
            # Start new station
            station_count += 1
            current_load = 0
            if any(times[t] > cycle_time for t in eligible):
                return 10**9
    
    return station_count

def _compute_critical_path(times, predecessors, successors, n):
    """Compute longest path from each task to any sink."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    if len(topo_order) != n:
        return None  # Cycle detected
    
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    
    return cp

def _simulate_annealing(priority, times, predecessors, cycle_time, n, deadline):
    """
    Simulated Annealing to improve the priority vector.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.95
    min_T = 0.01
    
    # We need to track time
    start_time = time.time()
    
    # Number of swaps to try per temperature step
    swaps_per_step = 5
    
    while T > min_T:
        for _ in range(swaps_per_step):
            if time.time() > deadline:
                return best_priority, best_score
            
            # Random move: swap two tasks
            i, j = random.sample(range(n), 2)
            
            # Save old values
            old_i, old_j = current_priority[i], current_priority[j]
            current_priority[i], current_priority[j] = old_j, old_i
            
            new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_priority = current_priority.copy()
                if best_score == 1:
                    return best_priority, best_score
            else:
                # Revert
                current_priority[i], current_priority[j] = old_i, old_j
        
        T *= cooling_rate
    
    return best_priority, best_score

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute features
    cp = _compute_critical_path(times, predecessors, successors, n)
    if cp is None:
        cp = np.zeros(n, dtype=float)
    
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Generate diverse initial heuristics
    heuristics = []
    
    # 1. Critical Path
    heuristics.append(cp.copy())
    
    # 2. Time + CP
    heuristics.append(times + cp)
    
    # 3. CP weighted by successors
    heuristics.append(cp + 0.5 * num_succ)
    
    # 4. Time weighted by CP
    heuristics.append(times * (cp / (np.max(cp) + 1e-6)))
    
    # 5. Pure Time
    heuristics.append(times.copy())
    
    # 6. CP - Preds (prioritize tasks that unblock more?)
    heuristics.append(cp - 0.1 * num_pred)
    
    # 7. Combined: 0.5*CP + 0.3*Time + 0.2*Successors
    heuristics.append(0.5 * cp + 0.3 * times + 0.2 * num_succ)
    
    # 8. Inverse Preds (prioritize tasks with fewer preds? No, usually high preds means late in graph)
    # Let's try: CP + Time - Successors (if a task has many successors, maybe it's a bottleneck, so HIGH priority)
    # So + Successors is correct.
    
    # 9. Rank-based: Sort by CP, assign rank
    rank_cp = np.argsort(np.argsort(cp))
    heuristics.append(rank_cp.astype(float))
    
    # Evaluate initial heuristics quickly
    best_priority = None
    best_score = float('inf')
    
    for h in heuristics:
        if time.time() > deadline - 20:
            break
        score = _greedy_balance(times, predecessors, cycle_time, h)
        if score < best_score:
            best_score = score
            best_priority = h.copy()
        if best_score == 1:
            break
    
    if best_priority is None:
        best_priority = heuristics[0]
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    
    # Run Simulated Annealing from the best initial
    if time.time() < deadline - 10:
        best_priority, best_score = _simulate_annealing(
            best_priority, times, predecessors, cycle_time, n, deadline - 5
        )
        
        # If time remains, try SA from a few other good heuristics
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 10 and best_score > 1:
            # Pick the next best heuristic
            scores = []
            for h in heuristics:
                scores.append(_greedy_balance(times, predecessors, cycle_time, h))
            # Sort heuristics by score
            sorted_indices = np.argsort(scores)
            
            # Try the 2nd and 3rd best
            for idx in sorted_indices[1:3]:
                if time.time() > deadline - 5:
                    break
                if scores[idx] >= best_score:
                    break
                pri, score = _simulate_annealing(
                    heuristics[idx], times, predecessors, cycle_time, n, deadline - 5
                )
                if score < best_score:
                    best_score = score
                    best_priority = pri
    
    return best_priority
# EVOLVE-BLOCK-END
