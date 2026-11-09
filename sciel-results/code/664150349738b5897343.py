"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def priority(instance):
    """
    Compute task priorities using a local search approach to minimize station count.
    Uses longest processing time as initial heuristic, then refines with local search.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    ct = instance.get("cycle_time", 40)
    
    # Build predecessor set for quick lookup
    preds = instance.get("predecessors", [])
    pred_sets = [set() for _ in range(n)]
    for i, p_list in enumerate(preds):
        for p in p_list:
            pred_sets[i].add(p)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    def simulate(priority_vec):
        """Simulate the greedy station filling algorithm. Returns number of stations."""
        remaining = set(range(n))
        completed = set()
        stations = 0
        
        while remaining:
            stations += 1
            station_time = 0.0
            
            # Find eligible tasks (all predecessors completed)
            eligible = []
            for i in remaining:
                if pred_sets[i].issubset(completed):
                    eligible.append(i)
            
            if not eligible:
                # No eligible tasks, but this shouldn't happen with valid precedence
                break
            
            # Sort eligible tasks by priority (descending)
            eligible_sorted = sorted(eligible, key=lambda i: -priority_vec[i])
            
            # Fill station with highest-priority tasks that fit
            for i in eligible_sorted:
                if station_time + times[i] <= ct + 1e-9:
                    station_time += times[i]
                    completed.add(i)
                    remaining.discard(i)
        
        return stations
    
    # Initial priority: longest processing time
    best_priority = times.copy()
    best_stations = simulate(best_priority)
    
    # Local search: try swapping pairs of priorities
    n_iterations = 0
    max_iterations = 50000
    
    while n_iterations < max_iterations:
        if time.time() > deadline:
            break
        
        n_iterations += 1
        
        # Pick two random tasks to swap
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap priorities
        old_i, old_j = best_priority[i], best_priority[j]
        best_priority[i], best_priority[j] = old_j, old_i
        
        new_stations = simulate(best_priority)
        
        if new_stations <= best_stations:
            best_stations = new_stations
            # Accept the swap
        else:
            # Revert the swap
            best_priority[i], best_priority[j] = old_i, old_j
    
    return best_priority
# EVOLVE-BLOCK-END
