import numpy as np

# Global state to track machine loads
_machine_loads = {}

def dispatch(ready_operations, time):
    global _machine_loads
    
    if not ready_operations:
        return 0

    # If this is the first call, initialize loads
    # We don't know the number of machines upfront, so we discover them from options
    # We need to ensure _machine_loads is reset for each new instance?
    # The problem says "five generated instances". The function is called repeatedly for one instance.
    # How to detect a new instance?
    # Usually, these evaluators call dispatch for a sequence of events for one instance, then start a new one.
    # We can detect a new instance if `time` resets to 0 or if the set of jobs changes drastically.
    # Safer: We can't easily reset. But if `time` is 0 and we have previous loads, it might be a new instance.
    # Let's assume the evaluator runs instances sequentially.
    # If time == 0, we should probably reset loads if we have seen time > 0 before.
    
    # Actually, a simpler heuristic that doesn't rely on fragile global state reset:
    # Use the "Critical Path" approximation based on remaining work and processing times.
    
    # Let's try a different distinct approach: **Weighted Shortest Job First (WSJF)**
    # Priority = remaining_work / total_duration_of_op
    # This is different from LRPT.
    
    # Let's go back to Global State but handle reset.
    # If time == 0, and we have non-zero loads, assume new instance and reset.
    if time == 0:
        if _machine_loads and any(v > 0 for v in _machine_loads.values()):
            _machine_loads = {}
        elif not _machine_loads:
            pass # First call

    best_idx = -1
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # Find the best machine for this operation based on current load
        # We want to assign it to the machine that will finish it earliest
        min_finish_time = float('inf')
        chosen_machine = None
        
        for machine, duration in options.items():
            current_load = _machine_loads.get(machine, 0)
            finish_time = current_load + duration
            if finish_time < min_finish_time:
                min_finish_time = finish_time
                chosen_machine = machine
                
        # Score: We want to minimize the "critical path" contribution.
        # A simple proxy is the finish time itself.
        # To avoid starvation, we can add a small penalty for jobs with high remaining work?
        # Or just pure List Scheduling.
        
        # Let's use a combined score:
        # 1. Primary: Minimize finish_time (Earliest Finish)
        # 2. Secondary: Minimize remaining_work (to finish jobs quickly)
        
        # Actually, minimizing finish_time tends to pick short ops on free machines.
        # This is good for throughput.
        
        score = min_finish_time
        
        # Tie breaker: Lower remaining work is better (SPT-like for jobs)
        # If scores are close, pick the job with less remaining work
        if score < best_score - 1e-9:
            best_score = score
            best_idx = i
        elif abs(score - best_score) < 1e-9:
            if op["remaining_work"] < ready_operations[best_idx]["remaining_work"]:
                best_idx = i

    if best_idx == -1:
        return 0

    # Update global state with the chosen operation's machine load
    chosen_op = ready_operations[best_idx]
    # Re-determine the chosen machine (same logic as above)
    options = chosen_op["machine_options"]
    min_finish_time = float('inf')
    chosen_machine = None
    for machine, duration in options.items():
        current_load = _machine_loads.get(machine, 0)
        finish_time = current_load + duration
        if finish_time < min_finish_time:
            min_finish_time = finish_time
            chosen_machine = machine
            
    _machine_loads[chosen_machine] = min_finish_time
    
    return int(best_idx)
