"""
Strategy: Local Search over Priority Vectors.
We treat the priority vector as the search space. We start with a heuristic vector
and iteratively improve it by perturbing scores. We simulate the evaluator's greedy
algorithm to evaluate each candidate. This is a meta-heuristic approach distinct
from direct assignment search (B&B).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _simulate_greedy_fast(instance, scores):
    """
    Fast simulation of the greedy algorithm.
    Returns number of stations.
    """
    times = instance["times"]
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    assigned = [False] * n_tasks
    station_count = 0
    current_load = 0.0
    
    # Precompute in-degrees for quick eligibility check
    # But precedence depends on assignment, so we need to update dynamically.
    # We'll just scan, but break early if possible?
    # For 30 tasks, O(N^2) is 900 ops, very fast.
    
    while any(not assigned[i] for i in range(n_tasks)):
        # Find best eligible task for current station
        best_task = -1
        best_score = -np.inf
        
        for t in range(n_tasks):
            if assigned[t]:
                continue
            # Check precedence
            ok = True
            for p in preds[t]:
                if p >= 0 and not assigned[p]:
                    ok = False
                    break
            if not ok:
                continue
            # Check fit
            if current_load + times[t] > cycle_time:
                continue
            if scores[t] > best_score:
                best_score = scores[t]
                best_task = t
                
        if best_task == -1:
            # No task fits in current station, or no eligible tasks
            # If current_load > 0, station is closed.
            # If current_load == 0, we need to start a new station with any eligible task that fits CT.
            if current_load > 0:
                station_count += 1
                current_load = 0.0
                # Loop again to fill new station
                continue
            else:
                # Start new station, pick any eligible task that fits CT
                # (Scores don't matter for the first pick if we just want to start? 
                # No, the greedy picks highest score among eligible that fits CT)
                # But if we are starting a new station, we pick highest score.
                # If no task fits CT, error.
                for t in range(n_tasks):
                    if assigned[t]:
                        continue
                    ok = True
                    for p in preds[t]:
                        if p >= 0 and not assigned[p]:
                            ok = False
                            break
                    if not ok:
                        continue
                    if times[t] <= cycle_time:
                        if scores[t] > best_score:
                            best_score = scores[t]
                            best_task = t
                            
                if best_task == -1:
                    # Should not happen
                    return n_tasks
                station_count += 1
                assigned[best_task] = True
                current_load = times[best_task]
                continue
        else:
            assigned[best_task] = True
            current_load += times[best_task]
            
    # Count the last station
    if current_load > 0:
        station_count += 1
        
    return station_count

def _initial_scores(instance):
    """
    Generate initial priority scores.
    """
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n_tasks = len(times)
    
    succs = [[] for _ in range(n_tasks)]
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0:
                succs[p].append(i)
                
    in_deg = [len(p) for p in preds]
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo = []
    indeg = in_deg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
                
    height = [0.0] * n_tasks
    for u in reversed(topo):
        if not succs[u]:
            height[u] = times[u]
        else:
            height[u] = times[u] + max(height[v] for v in succs[u])
            
    scores = np.zeros(n_tasks)
    max_h = max(height) if max(height) > 0 else 1.0
    for i in range(n_tasks):
        scores[i] = height[i] / max_h
        
    return scores

def priority(instance):
    """
    Local search over priority vectors.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    times = np.array(instance["times"], dtype=float)
    n_tasks = len(times)
    
    # Initial scores
    current_scores = _initial_scores(instance)
    current_stations = _simulate_greedy_fast(instance, current_scores)
    
    best_scores = current_scores.copy()
    best_stations = current_stations
    
    # Local search
    # We use a simple hill climbing with random restarts.
    # Move: pick two random tasks, swap their scores.
    # This preserves the set of scores but changes the ordering, which is what matters for greedy.
    
    search_deadline = start_time + 140
    
    while time.time() < search_deadline:
        # Check for restart
        # If we haven't improved in a while, restart
        # Simple heuristic: restart every 1000 iterations or if stuck
        
        # Pick two random distinct tasks
        t1 = random.randint(0, n_tasks - 1)
        t2 = random.randint(0, n_tasks - 1)
        while t2 == t1:
            t2 = random.randint(0, n_tasks - 1)
            
        # Swap scores
        current_scores[t1], current_scores[t2] = current_scores[t2], current_scores[t1]
        
        # Evaluate
        new_stations = _simulate_greedy_fast(instance, current_scores)
        
        if new_stations < current_stations:
            current_stations = new_stations
            if new_stations < best_stations:
                best_stations = new_stations
                best_scores = current_scores.copy()
        else:
            # Revert swap
            current_scores[t1], current_scores[t2] = current_scores[t2], current_scores[t1]
            
            # Occasional restart
            if random.random() < 0.01:
                current_scores = _initial_scores(instance)
                # Add noise
                for i in range(n_tasks):
                    current_scores[i] += random.gauss(0, 0.1)
                current_stations = _simulate_greedy_fast(instance, current_scores)
                
    return best_scores

# EVOLVE-BLOCK-END
