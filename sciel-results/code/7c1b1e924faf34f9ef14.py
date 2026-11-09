"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _simulate_greedy(instance, priorities):
    """
    Simulates the greedy station filling process.
    Returns the number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Build successor map for quick dependency checking
    # preds[i] is a list of tasks that must be done before i
    successors = [[] for _ in range(n_tasks)]
    indegree = [0] * n_tasks
    for i in range(n_tasks):
        for p in preds[i]:
            successors[p].append(i)
            indegree[i] += 1
            
    # Initial eligible tasks: those with no predecessors
    eligible = deque([i for i in range(n_tasks) if indegree[i] == 0])
    
    # We need to pick the highest priority eligible task that fits.
    # To do this efficiently, we maintain a list of eligible tasks.
    # Since n is small (30), we can just scan the eligible list each time.
    
    current_station_time = 0.0
    num_stations = 1
    assigned = [False] * n_tasks
    remaining_indegree = indegree.copy()
    
    # Copy eligible queue to a list for easier removal if needed, 
    # but since we only add and pick max, we can just keep a set/list.
    # Actually, the greedy rule is: "stations are filled with the highest-priority 
    # eligible task that fits the cycle time".
    
    # This implies a loop:
    # 1. Identify all eligible tasks (all preds assigned).
    # 2. Among those, pick the one with max priority that fits (current_time + task_time <= cycle_time).
    # 3. If none fits, start a new station.
    # 4. Assign task, update state.
    
    # Optimization: Pre-compute topological order? No, eligibility changes dynamically.
    
    while any(not assigned[i] for i in range(n_tasks)):
        # Find eligible tasks
        eligible_tasks = [i for i in range(n_tasks) if not assigned[i] and remaining_indegree[i] == 0]
        
        if not eligible_tasks:
            # This should not happen if the graph is a DAG
            break
            
        # Sort by priority descending
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
            # The loop will continue, and in the next iteration, it will pick a task
            # for the new station. Note: The task that didn't fit is still eligible.
            
    return num_stations

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Initial priority: Longest task first (standard heuristic)
    current_priorities = times.copy()
    best_score = _simulate_greedy(instance, current_priorities)
    best_priorities = current_priorities.copy()
    
    # Local Search: Simulated Annealing or Hill Climbing
    # We perturb the priorities and see if the station count improves.
    # Since station count is discrete and coarse, we might need many iterations.
    
    # Strategy: Randomly swap or perturb priorities of a few tasks.
    # Or, better: Randomly assign random priorities and keep the best.
    # Given 30 tasks, the search space is huge, but the objective (station count) 
    # has few distinct values.
    
    # Let's try a simple random restart with local improvement.
    # 1. Randomize priorities
    # 2. Evaluate
    # 3. If better, keep.
    # 4. Try small perturbations (swap two tasks' priorities) to improve.
    
    # To be more effective, we can use the "Rank Position Weight" (RPW) or similar,
    # but let's stick to direct optimization of the vector.
    
    # We will perform a series of random restarts.
    # Each restart:
    #  - Initialize with random priorities (or heuristic)
    #  - Perform local search (swap neighbors in priority order)
    
    # Actually, a simpler and often effective approach for this specific greedy algorithm:
    # The greedy algorithm's behavior is highly sensitive to the relative order of eligible tasks.
    # We can treat the priority vector as a permutation (if we assume distinct priorities) 
    # or a continuous vector.
    
    # Let's use a continuous vector and random perturbations.
    
    current_priorities = times.copy()
    best_score = _simulate_greedy(instance, current_priorities)
    best_priorities = current_priorities.copy()
    
    # Try to improve by randomizing and local search
    for restart in range(50): # Max 50 restarts
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        p = np.random.rand(n_tasks)
        # Scale to be comparable to times? Not strictly necessary for ordering.
        # But let's mix time and random to keep some structure.
        # p = times * 0.5 + np.random.rand(n_tasks) * 0.5
        # Let's just use random for diversity, or rank-based.
        
        score = _simulate_greedy(instance, p)
        if score < best_score:
            best_score = score
            best_priorities = p.copy()
        
        # Local search: try swapping priorities of two random tasks
        # or slightly perturbing.
        # Swapping is more discrete and might jump over plateaus.
        for i in range(200): # 200 steps per restart
            if time.time() > deadline - 5.0:
                break
                
            if i == 0:
                current_p = p.copy()
            else:
                # Perturb current best of this restart
                pass
            
            # Pick two random tasks
            idx1, idx2 = random.sample(range(n_tasks), 2)
            
            # Swap their priorities
            current_p[idx1], current_p[idx2] = current_p[idx2], current_p[idx1]
            
            new_score = _simulate_greedy(instance, current_p)
            
            if new_score <= best_score:
                best_score = new_score
                best_priorities = current_p.copy()
                p = current_p.copy() # Update the base for this restart
            else:
                # Revert
                current_p[idx1], current_p[idx2] = current_p[idx2], current_p[idx1]
                
    # Return the best priority vector found
    return best_priorities
# EVOLVE-BLOCK-END
