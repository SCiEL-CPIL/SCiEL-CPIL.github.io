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
    
    while eligible:
        # Sort eligible by priority descending
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        
        for task in candidates:
            if times[task] <= cycle_time - current_load:
                # Place task
                current_load += times[task]
                eligible.discard(task)
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

def _compute_downstream_load(times, predecessors, successors, n):
    """Compute total time of all successors (including self) for each task."""
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
        return np.zeros(n, dtype=float)
    
    dl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        dl[u] = times[u]
        for v in successors[u]:
            dl[u] += dl[v]
    
    return dl

def _compute_transitive_successors_count(successors, n):
    """Count number of transitive successors for each task."""
    trans_count = np.zeros(n, dtype=int)
    for i in range(n):
        # BFS/DFS from i
        visited = set()
        stack = list(successors[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for succ in successors[node]:
                    stack.append(succ)
        trans_count[i] = len(visited)
    return trans_count

def _compute_bottleneck_score(times, predecessors, successors, n):
    """
    Compute a bottleneck score based on the concept of minimum cut capacity.
    A task is a bottleneck if delaying it would delay many successors.
    We approximate this by combining critical path with successor structure.
    """
    cp = _compute_critical_path(times, predecessors, successors, n)
    if cp is None:
        cp = np.zeros(n, dtype=float)
    
    dl = _compute_downstream_load(times, predecessors, successors, n)
    
    # Number of direct successors
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of transitive successors
    trans_succ = _compute_transitive_successors_count(successors, n).astype(float)
    
    # Normalize
    cp_norm = cp / (np.max(cp) + 1e-6)
    dl_norm = dl / (np.max(dl) + 1e-6)
    succ_norm = num_succ / (np.max(num_succ) + 1e-6)
    trans_norm = trans_succ / (np.max(trans_succ) + 1e-6)
    
    # Bottleneck score: high CP + high downstream load + many successors (direct + transitive)
    bottleneck = 0.35 * cp_norm + 0.30 * dl_norm + 0.15 * succ_norm + 0.20 * trans_norm
    
    return bottleneck, cp, dl

def _local_search(priority, times, predecessors, cycle_time, n, deadline):
    """
    Local search: try all pairwise swaps and keep the best improvement.
    Repeat until no improvement or deadline.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return best_priority, best_score
            for j in range(i+1, n):
                # Try swap
                old_i, old_j = current_priority[i], current_priority[j]
                current_priority[i], current_priority[j] = old_j, old_i
                
                new_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_priority = current_priority.copy()
                    improved = True
                    if best_score == 1:
                        return best_priority, best_score
                else:
                    # Revert
                    current_priority[i], current_priority[j] = old_i, old_j
    
    return best_priority, best_score

def _simulate_annealing(priority, times, predecessors, cycle_time, n, deadline):
    """
    Simulated Annealing to improve the priority vector.
    Uses swap moves and occasional insertion moves for better exploration.
    """
    current_priority = priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score
    
    # Temperature schedule
    T = 1.0
    cooling_rate = 0.96
    min_T = 0.005
    
    start_time = time.time()
    
    # Number of swaps to try per temperature step
    swaps_per_step = 8
    
    while T > min_T:
        for _ in range(swaps_per_step):
            if time.time() > deadline:
                return best_priority, best_score
            
            # Random move: 80% swap, 20% perturbation (add noise to two random tasks)
            if random.random() < 0.8:
                i, j = random.sample(range(n), 2)
                old_i, old_j = current_priority[i], current_priority[j]
                current_priority[i], current_priority[j] = old_j, old_i
            else:
                i, j = random.sample(range(n), 2)
                old_i, old_j = current_priority[i], current_priority[j]
                current_priority[i] += random.gauss(0, 0.5)
                current_priority[j] += random.gauss(0, 0.5)
            
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
    deadline = start_time + 130.0  # Safety margin
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute features
    bottleneck, cp, dl = _compute_bottleneck_score(times, predecessors, successors, n)
    
    num_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    num_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    trans_succ = _compute_transitive_successors_count(successors, n).astype(float)
    
    # Generate diverse initial heuristics
    heuristics = []
    
    # 1. Critical Path
    heuristics.append(cp.copy())
    
    # 2. Time + CP
    heuristics.append(times + cp)
    
    # 3. Downstream Load
    heuristics.append(dl.copy())
    
    # 4. Bottleneck score (new)
    heuristics.append(bottleneck.copy())
    
    # 5. CP weighted by successors
    heuristics.append(cp + 0.5 * num_succ)
    
    # 6. Time weighted by CP
    heuristics.append(times * (cp / (np.max(cp) + 1e-6)))
    
    # 7. Pure Time
    heuristics.append(times.copy())
    
    # 8. CP - Preds
    heuristics.append(cp - 0.1 * num_pred)
    
    # 9. Combined: 0.5*CP + 0.3*Time + 0.2*Successors
    heuristics.append(0.5 * cp + 0.3 * times + 0.2 * num_succ)
    
    # 10. Rank-based: Sort by CP, assign rank
    rank_cp = np.argsort(np.argsort(cp))
    heuristics.append(rank_cp.astype(float))
    
    # 11. Downstream Load + Time
    heuristics.append(dl + times)
    
    # 12. CP * Successors (bottleneck awareness)
    heuristics.append(cp * (1 + 0.1 * num_succ))
    
    # 13. Bottleneck + Time
    heuristics.append(bottleneck + 0.5 * times)
    
    # 14. Bottleneck * CP
    heuristics.append(bottleneck * cp)
    
    # 15. CP + transitive successors
    heuristics.append(cp + 0.3 * trans_succ)
    
    # 16. Downstream load + transitive successors
    heuristics.append(dl + 0.2 * trans_succ)
    
    # Evaluate initial heuristics quickly
    best_priority = None
    best_score = float('inf')
    
    for h in heuristics:
        if time.time() > deadline - 30:
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
    
    # Run Local Search first (more focused, often finds good local optima quickly)
    if time.time() < deadline - 15:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n, deadline - 10
        )
    
    # Run Simulated Annealing from the best result
    if time.time() < deadline - 5:
        best_priority, best_score = _simulate_annealing(
            best_priority, times, predecessors, cycle_time, n, deadline - 2
        )
        
        # If time remains, try SA from a few other good heuristics
        remaining_time = deadline - 2 - time.time()
        if remaining_time > 8 and best_score > 1:
            # Pick the next best heuristic
            scores = []
            for h in heuristics:
                scores.append(_greedy_balance(times, predecessors, cycle_time, h))
            # Sort heuristics by score
            sorted_indices = np.argsort(scores)
            
            # Try the 2nd and 3rd best
            for idx in sorted_indices[1:3]:
                if time.time() > deadline - 2:
                    break
                if scores[idx] >= best_score:
                    break
                pri, score = _simulate_annealing(
                    heuristics[idx], times, predecessors, cycle_time, n, deadline - 2
                )
                if score < best_score:
                    best_score = score
                    best_priority = pri
    
    return best_priority
# EVOLVE-BLOCK-END
