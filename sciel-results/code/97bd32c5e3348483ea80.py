import numpy as np
import time

def _simulate(priority_scores, times, preds, cycle_time):
    n = len(times)
    assigned = [False] * n
    stations = 0
    current_time = 0
    
    # Pre-check: if any time > cycle_time, return infinity or large number
    for t in times:
        if t > cycle_time:
            return 10000

    while True:
        # Check if all assigned
        if all(assigned):
            break
            
        # Find dynamic available
        available = []
        for i in range(n):
            if not assigned[i]:
                if all(assigned[p] for p in preds[i]):
                    available.append(i)
                        
        if not available:
            break # Deadlock or error
            
        # Sort by priority (descending)
        available.sort(key=lambda x: priority_scores[x], reverse=True)
        
        placed_in_this_step = False
        for i in available:
            if times[i] + current_time <= cycle_time:
                assigned[i] = True
                current_time += times[i]
                placed_in_this_step = True
                break # Pick ONLY the highest priority that fits, then re-evaluate.
                
        if not placed_in_this_step:
            # No task fits in current station
            if current_time > 0:
                stations += 1
                current_time = 0
            else:
                # Empty station, no task fits?
                # This implies the highest priority available task doesn't fit in an empty station.
                # i.e., times[i] > cycle_time.
                # Handled by pre-check.
                break

    if current_time > 0:
        stations += 1
        
    return stations

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Calculate RPV
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            
    rpv = np.zeros(n)
    for i in range(n):
        for j in succs[i]:
            rpv[i] += times[j]
            
    # Normalize
    max_t = np.max(times) if np.max(times) > 0 else 1
    max_r = np.max(rpv) if np.max(rpv) > 0 else 1
    
    times_norm = times / max_t
    rpv_norm = rpv / max_r
        
    # Initial Score: Time + RPV
    best_scores = times_norm + rpv_norm
    best_val = _simulate(best_scores, times, preds, cycle_time)
    
    # Hill Climbing
    start_time = time.time()
    deadline = 100.0 # 100s limit
    
    no_improve = 0
    max_no_improve = 1000
    
    # Use a fixed seed for reproducibility? 
    # np.random.seed(42)
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Perturb: Swap two random scores
        i, j = np.random.choice(n, 2, replace=False)
        
        # Create new scores
        new_scores = best_scores.copy()
        new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
        
        val = _simulate(new_scores, times, preds, cycle_time)
        
        if val < best_val:
            best_val = val
            best_scores = new_scores
            no_improve = 0
        else:
            no_improve += 1
            
    return best_scores
