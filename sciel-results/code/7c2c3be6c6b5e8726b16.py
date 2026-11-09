"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _simulate_greedy(instance, priorities, collect_slack=False):
    """
    Simulates the greedy station filling process.
    Returns the number of stations used.
    If collect_slack=True, also returns a vector of "slack values" observed during simulation.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Build successor map for quick dependency checking
    successors = [[] for _ in range(n_tasks)]
    indegree = [0] * n_tasks
    for i in range(n_tasks):
        for p in preds[i]:
            successors[p].append(i)
            indegree[i] += 1
            
    current_station_time = 0.0
    num_stations = 1
    assigned = [False] * n_tasks
    remaining_indegree = indegree.copy()
    
    # For slack collection: track how "tight" each task was when placed
    # slack[task] = cycle_time - (time_before_task + time_of_task)
    # Lower slack means tighter fit. We want to prioritize tasks with low slack.
    slack_accum = np.zeros(n_tasks)
    slack_count = np.zeros(n_tasks)
    
    while any(not assigned[i] for i in range(n_tasks)):
        # Find eligible tasks
        eligible_tasks = [i for i in range(n_tasks) if not assigned[i] and remaining_indegree[i] == 0]
        
        if not eligible_tasks:
            break
            
        # Sort by priority descending
        eligible_tasks.sort(key=lambda x: priorities[x], reverse=True)
        
        # Find the first one that fits
        found = False
        for task in eligible_tasks:
            if current_station_time + times[task] <= cycle_time:
                # Assign task
                assigned[task] = True
                
                if collect_slack:
                    # Record the slack for this task at this placement
                    slack_before = cycle_time - current_station_time
                    slack_after = cycle_time - (current_station_time + times[task])
                    # The "tightness" is how close we got to filling the station
                    # We'll use the remaining slack after placing the task
                    slack_accum[task] += slack_after
                    slack_count[task] += 1
                
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
            
    if collect_slack:
        # Average slack per task (only for tasks that were placed)
        avg_slack = np.zeros(n_tasks)
        for i in range(n_tasks):
            if slack_count[i] > 0:
                avg_slack[i] = slack_accum[i] / slack_count[i]
            else:
                avg_slack[i] = cycle_time
        return num_stations, avg_slack
    else:
        return num_stations

def _compute_slack_priorities(instance, base_priorities):
    """
    Compute priority vector based on observed slack values.
    Tasks with lower average slack (tighter fits) get higher priority.
    We combine this with the base priorities (e.g., LPT) for stability.
    """
    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # Run simulation to collect slack
    _, avg_slack = _simulate_greedy(instance, base_priorities, collect_slack=True)
    
    # Normalize slack to [0, 1]
    min_slack = np.min(avg_slack)
    max_slack = np.max(avg_slack)
    if max_slack > min_slack:
        norm_slack = (avg_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.zeros(n_tasks)
    
    # Lower slack = tighter fit = higher priority
    # So priority should be inversely related to slack
    slack_priority = 1.0 - norm_slack
    
    # Combine with time-based priority (LPT)
    min_time = np.min(times)
    max_time = np.max(times)
    if max_time > min_time:
        norm_time = (times - min_time) / (max_time - min_time)
    else:
        norm_time = np.zeros(n_tasks)
    
    # Weighted combination: 70% slack-based, 30% time-based
    combined = 0.7 * slack_priority + 0.3 * norm_time
    
    return combined

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
    
    # Try slack-based priorities
    slack_p = _compute_slack_priorities(instance, current_priorities)
    slack_score = _simulate_greedy(instance, slack_p)
    if slack_score < best_score:
        best_score = slack_score
        best_priorities = slack_p.copy()
    
    # Local Search: Hill Climbing with slack-guided initialization
    # We'll use a combination of:
    # 1. Random restarts with slack-based seeding
    # 2. Local search via swaps
    
    for restart in range(50): # Max 50 restarts
        if time.time() > deadline - 5.0:
            break
            
        # Initialization: mix of slack-based and random
        if restart % 3 == 0:
            # Pure slack-based
            p = _compute_slack_priorities(instance, best_priorities)
        elif restart % 3 == 1:
            # Slack + random noise
            p = _compute_slack_priorities(instance, best_priorities)
            p = p + 0.1 * np.random.randn(n_tasks)
        else:
            # Pure random for diversity
            p = np.random.rand(n_tasks)
        
        score = _simulate_greedy(instance, p)
        if score < best_score:
            best_score = score
            best_priorities = p.copy()
        
        # Local search: try swapping priorities of two random tasks
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
                
    # Final polish: try slack-based refinement on the best found
    if time.time() < deadline - 10.0:
        polished = _compute_slack_priorities(instance, best_priorities)
        pol_score = _simulate_greedy(instance, polished)
        if pol_score < best_score:
            best_score = pol_score
            best_priorities = polished.copy()
    
    # Return the best priority vector found
    return best_priorities
# EVOLVE-BLOCK-END
