"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque

def _simulate_greedy(instance, priorities):
    """
    Simulates the greedy station filling process.
    Returns the number of stations used.
    """
    times = instance["_times"]
    successors = instance["_successors"]
    indegree_init = instance["_indegree"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Optimization: Pre-calculate initial eligible tasks
    # But since indegree changes, we just scan. n=30 is small.
    
    current_station_time = 0.0
    num_stations = 1
    assigned = [False] * n_tasks
    remaining_indegree = indegree_init.copy()
    
    while True:
        # Check if all assigned
        # Optimization: Count assigned? Or just check any?
        # any() is fast enough.
        if all(assigned):
            break
            
        # Find eligible tasks
        # Optimization: Instead of scanning all 30, we could maintain a list.
        # But scanning 30 is very fast.
        eligible_tasks = [i for i in range(n_tasks) if not assigned[i] and remaining_indegree[i] == 0]
        
        if not eligible_tasks:
            # Deadlock or error
            break
            
        # Sort by priority descending
        # Optimization: Use sorted with key.
        eligible_tasks.sort(key=lambda x: priorities[x], reverse=True)
        
        # Find the first one that fits
        found = False
        for task in eligible_tasks:
            if current_station_time + times[task] <= cycle_time:
                # Assign task
                assigned[task] = True
                current_station_time += times[task]
                
                # Update successors
                for succ in successors[task]:
                    remaining_indegree[succ] -= 1
                found = True
                break
        
        if not found:
            # Start new station
            num_stations += 1
            current_station_time = 0.0
            # The loop continues. The same eligible tasks are available.
            # In the next iteration, since current_station_time is 0,
            # the highest priority task (which has time <= cycle_time) will fit.
            
    return num_stations

def _prepare_instance(instance):
    """
    Pre-computes data structures for fast simulation.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    successors = [[] for _ in range(n_tasks)]
    indegree = [0] * n_tasks
    for i in range(n_tasks):
        for p in preds[i]:
            successors[p].append(i)
            indegree[i] += 1
            
    return {
        "times": times,
        "successors": successors,
        "indegree": indegree,
        "cycle_time": cycle_time,
        "n_tasks": n_tasks
    }

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # Pre-compute static data
    prepared = _prepare_instance(instance)
    times = prepared["times"]
    n_tasks = prepared["n_tasks"]
    cycle_time = prepared["cycle_time"]
    
    # Lower bound
    lb = np.ceil(np.sum(times) / cycle_time)
    
    # Initial candidates
    # 1. Longest Task First
    h_ltf = times.copy()
    
    # 2. Rank Position Weight (RPW) approximation
    # Sort by time descending
    sorted_indices = np.argsort(-times)
    ranks = np.zeros(n_tasks)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = n_tasks - rank
    h_rpw = times * ranks
    
    # 3. Weighted (Time + Successors)
    succ_count = np.zeros(n_tasks, dtype=int)
    for i in range(n_tasks):
        for p in instance["predecessors"][i]:
            succ_count[p] += 1
    h_weighted = times + 0.1 * succ_count
            
    # 4. Random
    h_random = np.random.rand(n_tasks)
    
    candidates = [h_ltf, h_weighted, h_rpw, h_random]
    
    best_score = float('inf')
    best_priorities = None
    
    for cand in candidates:
        score = _simulate_greedy(prepared, cand)
        if score < best_score:
            best_score = score
            best_priorities = cand.copy()
            
    if best_score == lb:
        return best_priorities

    # Local Search
    # Strategy:
    # 1. Random restarts with hill climbing on swaps.
    # 2. Adaptive step size or neighborhood?
    # 3. Keep global best.
    
    # We can also try to "refine" the best solution by checking all pairs?
    # O(n^2) = 900 checks. Each check is a simulation.
    # 900 * 5ms = 4.5s.
    # We can do a full pairwise swap check on the best solution periodically.
    
    current_priorities = best_priorities.copy()
    
    num_restarts = 100
    steps_per_restart = 100
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current for this restart
        if restart == 0:
            p = best_priorities.copy()
        else:
            # Mix: 30% random, 70% perturbed best
            if random.random() < 0.3:
                p = np.random.rand(n_tasks)
            else:
                p = best_priorities + np.random.normal(0, 0.1, n_tasks)
                
        score = _simulate_greedy(prepared, p)
        if score < best_score:
            best_score = score
            best_priorities = p.copy()
            if best_score == lb:
                return best_priorities
                
        # Local Search: Hill Climbing
        for i in range(steps_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            # Pick two random tasks to swap
            idx1, idx2 = random.sample(range(n_tasks), 2)
            
            # Swap priorities
            p[idx1], p[idx2] = p[idx2], p[idx1]
            
            new_score = _simulate_greedy(prepared, p)
            
            if new_score <= best_score:
                best_score = new_score
                best_priorities = p.copy()
                if best_score == lb:
                    return best_priorities
            else:
                # Revert swap
                p[idx1], p[idx2] = p[idx2], p[idx1]
                
            # Occasionally, do a "full scan" improvement on the current best
            # to escape shallow local optima caused by random swaps.
            # If we have time, try all single swaps on best_priorities?
            # That's expensive. Maybe just every 10 steps, try a few more targeted swaps.
            # For now, stick to random swaps.
            
    # Final Polish:
    # If we have time, try to improve best_priorities with a more exhaustive local search.
    # e.g., try swapping every pair in best_priorities.
    if time.time() < deadline - 2.0:
        p = best_priorities.copy()
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(n_tasks):
                for j in range(i+1, n_tasks):
                    if time.time() > deadline - 2.0:
                        break
                    p[i], p[j] = p[j], p[i]
                    score = _simulate_greedy(prepared, p)
                    if score < best_score:
                        best_score = score
                        best_priorities = p.copy()
                        improved = True
                        if best_score == lb:
                            return best_priorities
                    else:
                        p[i], p[j] = p[j], p[i]
                if time.time() > deadline - 2.0:
                    break

    return best_priorities
# EVOLVE-BLOCK-END
