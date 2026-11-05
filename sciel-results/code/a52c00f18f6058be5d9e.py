"""Rewritten program with a search-based priority for ALB."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def _simulate_greedy(order, times, succ, pred, cycle_time):
    """
    Simulate the greedy packing process for a given priority order.
    Returns the number of stations used.
    """
    n = len(times)
    assigned = [False] * n
    # Initial available tasks
    available = [i for i in range(n) if len(pred[i]) == 0]
    
    # Rank array for quick lookup
    rank = [0] * n
    for idx, task in enumerate(order):
        rank[task] = idx
        
    stations = 0
    current_load = 0.0
    
    # To speed up, we can keep available as a list and filter.
    # N is small (30), so linear scan is fine.
    
    while sum(assigned) < n:
        best_task = -1
        best_rank = n + 1
        
        # Find the highest priority task that fits
        for t in available:
            if not assigned[t] and times[t] <= cycle_time - current_load + 1e-9:
                if rank[t] < best_rank:
                    best_rank = rank[t]
                    best_task = t
                    
        if best_task != -1:
            # Assign the task
            assigned[best_task] = True
            current_load += times[best_task]
            
            # Remove from available
            # List removal is O(N), but N is small.
            available.remove(best_task)
            
            # Update available with successors
            for s in succ[best_task]:
                # Check if all predecessors of s are assigned
                # Optimization: keep a count of unassigned preds
                # For now, simple check is O(N^3) total, which is fine for N=30.
                all_preds_assigned = True
                for p in pred[s]:
                    if not assigned[p]:
                        all_preds_assigned = False
                        break
                if all_preds_assigned and s not in available:
                    available.append(s)
        else:
            # No task fits in the current station
            if current_load > 1e-9:
                stations += 1
                current_load = 0.0
            else:
                # If load is 0 and no task fits, we are stuck.
                # This shouldn't happen in valid instances where max time <= cycle_time.
                # But if it does, break to avoid infinite loop.
                break

    if current_load > 1e-9:
        stations += 1
        
    return stations

def priority(instance):
    """
    Uses Simulated Annealing to find the best static priority order
    that minimizes the number of stations in the greedy packing.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
            
    # Initial Order: Based on Longest Path (descending)
    # Compute longest path to terminal
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) != n:
        # Cycle detected, fallback
        return times.copy()
        
    dist = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist[v])
        dist[u] = times[u] + max_succ
        
    # Initial order: sort by dist descending
    initial_order = sorted(range(n), key=lambda x: -dist[x])
    
    # Evaluate initial
    best_order = initial_order.copy()
    best_score = _simulate_greedy(best_order, times, succ, pred, cycle_time)
    
    # Simulated Annealing
    current_order = best_order.copy()
    current_score = best_score
    T = 1.0
    cooling_rate = 0.995
    
    # Estimate max iterations based on time
    # Each iteration is fast, ~10-50 microseconds.
    # Let's just loop until time is up.
    
    while time.time() < deadline:
        # Generate a new order by swapping two random elements
        new_order = current_order.copy()
        i, j = random.sample(range(n), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_score = _simulate_greedy(new_order, times, succ, pred, cycle_time)
        
        if new_score < current_score:
            current_order = new_order
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_order = current_order.copy()
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_score - current_score
            if random.random() < math.exp(-delta / T):
                current_order = new_order
                current_score = new_score
                
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor temperature to allow some exploration
            
    # Convert best_order to scores
    # Higher score = higher priority
    # Rank 0 in order = highest priority = highest score
    scores = np.zeros(n)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores
# EVOLVE-BLOCK-END
