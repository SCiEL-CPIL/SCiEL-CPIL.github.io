"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _compute_heights(n, preds, times):
    """Compute longest path from each node to the end (height)."""
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1
            
    q = deque([i for i in range(n) if indeg[i] == 0])
    h = [0.0] * n
    while q:
        u = q.popleft()
        for v in succs[u]:
            if h[u] + times[u] > h[v]:
                h[v] = h[u] + times[u]
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return h

def _greedy_count(n, times, preds, ct, order):
    """Simulate the evaluator's greedy assignment and return number of stations."""
    if not order:
        return 0
        
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            
    remaining = {i: 0 for i in range(n)}
    for i in range(n):
        for p in preds[i]:
            remaining[i] += 1
            
    done = [False] * n
    stations = []
    current_sum = 0.0
    count = 0
    
    for task in order:
        if done[task]:
            continue
            
        # Check if eligible (all predecessors done)
        eligible = True
        for p in preds[task]:
            if not done[p]:
                eligible = False
                break
        
        if not eligible:
            # In a valid topological order, this shouldn't happen if the order is valid.
            # But if the order is invalid (not topological), the greedy might skip it?
            # The task description says "eligible task that fits".
            # If the order is not topological, the greedy might get stuck or behave differently.
            # However, standard ALB assumes the priority list is a topological sort.
            # If we generate random permutations, they might not be topological.
            # The evaluator likely expects the priority scores to induce a valid ordering.
            # If we return scores that don't correspond to a valid topological sort,
            # the evaluator's behavior is undefined or it might fail.
            # To be safe, we should ensure our search space consists of valid topological sorts.
            # For now, if we hit this, we can't place it. But the evaluator's greedy
            # "fills with highest priority eligible task". It scans the list.
            # If a task is not eligible, it is skipped.
            # So if our order has a task before its predecessor, the evaluator will skip it
            # until the predecessor is placed.
            # This effectively means the evaluator performs a stable topological sort based on priorities.
            # So our simulation must mimic this: iterate through the order, and only place
            # if eligible.
            continue

        if current_sum + times[task] <= ct:
            current_sum += times[task]
            done[task] = True
            for s in succs[task]:
                remaining[s] -= 1
        else:
            # Close current station
            count += 1
            stations.append(current_sum)
            current_sum = times[task]
            done[task] = True
            for s in succs[task]:
                remaining[s] -= 1
            
    if current_sum > 0:
        count += 1
        
    return count

def priority(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    ct = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # 1. Compute initial valid topological order using Longest Processing Time (LPT) heuristic
    # This serves as our initial best solution.
    
    # We need to generate a topological sort.
    # We can do this by repeatedly picking the available task with the longest time.
    # This is a standard heuristic.
    
    succs = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            indeg[i] += 1
            
    # Initial LPT Topological Sort
    available = [i for i in range(n) if indeg[i] == 0]
    done = [False] * n
    order_lpt = []
    
    # To make it deterministic and fast for initialization
    while available:
        # Pick max time
        max_idx = -1
        max_time = -1
        for i in available:
            if times[i] > max_time:
                max_time = times[i]
                max_idx = i
        if max_idx == -1: break
        order_lpt.append(max_idx)
        done[max_idx] = True
        for s in succs[max_idx]:
            indeg[s] -= 1
            if indeg[s] == 0:
                available.append(s)
                
    # If the graph is not a DAG, this will fail, but we assume valid input.
    # If order_lpt is shorter than n, there's a cycle. We assume no cycles.
    
    # Simulate to get initial score
    best_order = order_lpt
    best_stations = _greedy_count(n, times, preds, ct, best_order)
    
    # 2. Simulated Annealing
    # State: a valid topological order.
    # Move: Swap two elements in the order if the resulting sequence is still a valid topological sort.
    # This is expensive to check. 
    # Alternative: Generate a new random topological sort by perturbing the priorities?
    # Or: Use a "bubble" move: swap adjacent elements if valid.
    
    # Let's try a simpler local search: 
    # Randomly pick two positions i, j in the order.
    # Swap them.
    # Check if the new order is a valid topological sort.
    # If yes, evaluate. If better, accept. If worse, accept with prob exp(-delta/T).
    
    # Checking validity of a topological sort is O(n^2) or O(n+E).
    # With n=30, O(n^2) is 900 ops. Very fast.
    
    def is_valid_topo(order, preds, succs):
        pos = {task: i for i, task in enumerate(order)}
        for i in range(n):
            for p in preds[i]:
                if pos[p] > pos[i]:
                    return False
        return True

    # Initialize random state
    current_order = best_order[:]
    current_stations = best_stations
    
    # Temperature schedule
    T = 10.0
    cooling = 0.995
    
    # We need to generate neighbors.
    # A random swap might often be invalid.
    # A better move: Pick a random task, and move it earlier or later by 1 step, if valid.
    # Or: Pick two random tasks, try to swap.
    
    # Let's use a "random restart" or "random perturbation" strategy if stuck?
    # For now, simple SA with swap.
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Generate neighbor
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        if not is_valid_topo(new_order, preds, succs):
            continue
            
        new_stations = _greedy_count(n, times, preds, ct, new_order)
        
        delta = new_stations - current_stations
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_order = new_order
            current_stations = new_stations
            
            if current_stations < best_stations:
                best_stations = current_stations
                best_order = current_order[:]
                
        T *= cooling
        
    # 3. Convert best_order to priority scores
    # The evaluator sorts by score descending.
    # So the first element in best_order should have the highest score.
    # score[task] = n - index_of_task_in_best_order
    
    scores = np.zeros(n, dtype=float)
    for idx, task in enumerate(best_order):
        scores[task] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
