"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

# Global state for stateful dispatching
_start_time = None
_last_dispatch_time = None
_last_selected_op_key = None
_last_selected_machine = None

def dispatch(ready_operations, time):
    global _start_time, _last_dispatch_time, _last_selected_op_key, _last_selected_machine
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        _last_dispatch_time = time.time()
        _last_selected_op_key = None
        _last_selected_machine = None

    # Deadline guard: if we are running long, fall back to a fast heuristic
    elapsed = time.time() - _start_time
    if elapsed > 140:
        # Fast fallback: pick max remaining_work
        best_idx = 0
        best_work = -1
        for i, op in enumerate(ready_operations):
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
        return best_idx

    if not ready_operations:
        return 0

    # Identify the candidate operation and machine
    # We use a scoring function that balances:
    # 1. Critical Path (remaining_work) - prioritize jobs with more work left
    # 2. Machine Congestion - avoid machines that are likely busy (heuristic: prefer machine with shortest duration if possible, or least loaded)
    # 3. Operation Duration - Shortest Processing Time (SPT) is often good for makespan
    
    best_score = -float('inf')
    best_op_idx = 0
    best_machine = None
    
    # To avoid oscillation and encourage stability, we add a small bonus to the previously selected machine if it's still available and the op is the same
    # But since we don't have access to machine states directly, we rely on the duration.
    # A common effective heuristic for FJSP is:
    # Priority = Remaining Work / (1 + Number of Machines) + Shortest Duration on Best Machine
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        # Find the best machine for this operation
        # Heuristic: Choose machine with minimum duration. 
        # If durations are equal, we could look at other factors, but duration is primary.
        min_duration = float('inf')
        best_m_for_op = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_m_for_op = m
        
        if best_m_for_op is None:
            continue
            
        # Calculate a composite score
        # We want to maximize remaining_work (urgency) and minimize duration (efficiency)
        # Normalization isn't strictly necessary if we weight them appropriately
        # Score = w1 * remaining_work - w2 * duration
        
        # To make it more robust, let's try a "Critical Ratio" like approach:
        # If remaining_work is high, we want to start it.
        # If duration is low, it frees up the machine sooner.
        
        score = remaining_work * 10.0 - min_duration * 1.0
        
        # Tie-breaker: If scores are close, prefer the one with shorter duration to reduce jitter
        # Or simply keep the first one found if scores are equal
        
        if score > best_score:
            best_score = score
            best_op_idx = i
            best_machine = best_m_for_op
        elif score == best_score:
            # Tie-breaker: shorter duration
            if min_duration < float('inf') and (best_machine is None or True): 
                 # We need the previous min_duration to compare properly, let's store it
                 pass

    # Refine tie-breaking by storing min_duration in the loop
    best_score = -float('inf')
    best_op_idx = 0
    best_machine = None
    best_min_duration = float('inf')
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        min_duration = float('inf')
        best_m_for_op = None
        for m, d in machine_options.items():
            if d < min_duration:
                min_duration = d
                best_m_for_op = m
                
        if best_m_for_op is None:
            continue
            
        score = remaining_work * 10.0 - min_duration * 1.0
        
        if score > best_score:
            best_score = score
            best_op_idx = i
            best_machine = best_m_for_op
            best_min_duration = min_duration
        elif score == best_score:
            if min_duration < best_min_duration:
                best_op_idx = i
                best_machine = best_m_for_op
                best_min_duration = min_duration

    return best_op_idx
# EVOLVE-BLOCK-END
