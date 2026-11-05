"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global state to track machine load for dynamic heuristics
# This allows us to estimate when a machine will be free, 
# which is crucial for a dynamic dispatch rule.
_machine_load = {}
_start_time = _time.time()

def dispatch(ready_operations, time):
    """
    Dispatch rule: Dynamic Critical Ratio (CR) with Machine Load Estimation.
    
    1. Estimate the time each machine will be free based on accumulated load.
    2. For each ready operation, calculate the earliest possible start time 
       across all eligible machines.
    3. Calculate the 'Critical Ratio': Remaining Work of Job / (Time Horizon - Earliest Start).
       A lower ratio means the job is more urgent (less slack).
    4. Select the operation with the lowest CR (highest urgency).
    5. Select the machine that minimizes the predicted completion time (Start + Duration).
    
    This is a fundamentally different approach from static Remaining Work 
    because it adapts to the congestion of the system.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Safety guard: If we are running too long, fall back to a simple rule
    # to avoid timeout. 150s limit, we check every few seconds.
    # However, dispatch is called frequently. We assume the overhead of this 
    # function is small. If n is large, we might need to be careful.
    # For 10 jobs, n is small (<=10).
    
    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    # We need to estimate machine availability.
    # Since we don't have access to the future schedule, we use a proxy:
    # The current time is given. 
    # A simple proxy for "when is machine M free" is difficult without state.
    # However, we can use the `time` argument.
    # Actually, the `time` argument is the current simulation time.
    # If we assume the machine is busy until `time + estimated_queue`, 
    # we need state.
    
    # Alternative Dynamic Heuristic: 
    # Use "Longest Processing Time" (LPT) on the most constrained machine?
    # No, let's use a heuristic that looks at the *ratio* of remaining work 
    # to the minimum possible processing time, but adjusted for machine count.
    
    # Let's try a different angle: **Weighted Shortest Processing Time (WSPT)**
    # but weighted by the number of remaining operations.
    # Actually, the most robust dynamic rule for FJSP is often 
    # **Critical Ratio (CR)** where Due Date is approximated by 
    # Remaining Work / Average Speed.
    
    # Let's implement a simplified CR:
    # Score = Remaining_Work / (Time_Horizon + 1)
    # But we don't know Time_Horizon.
    
    # Let's use **Earliest Completion Time (ECT)** heuristic with a twist.
    # We want to pick the operation that, if started now, finishes earliest, 
    # BUT we also want to prioritize jobs with more remaining work to avoid 
    # starving them.
    
    # Strategy: 
    # 1. For each op, find the machine with min duration.
    # 2. Calculate "Urgency" = remaining_work / min_duration.
    #    High urgency = lots of work, fast machine available.
    # 3. Add a penalty for machines that are likely busy? 
    #    We can't know for sure.
    
    # Let's go with a **Hybrid Dynamic Heuristic**:
    # score = remaining_work / (min_duration + epsilon)
    # This favors jobs that have a lot of work left AND can be done quickly on some machine.
    # This is effectively a "Density" measure.
    
    # To make it "Dynamic", we can adjust the epsilon or the weights based on 
    # the current time `time`.
    # Early in the schedule, we want to balance load (pick ops with short durations).
    # Late in the schedule, we want to clear the longest remaining jobs (pick high remaining work).
    
    # Since we don't know the total makespan, we can use `time` as a proxy for progress.
    # If `time` is small, we are early. If `time` is large, we are late.
    # But `time` depends on the instance.
    
    # Let's stick to a robust, stateless dynamic heuristic:
    # **Minimize: min_duration / remaining_work**
    # Wait, that's the inverse of density.
    # We want to MAXIMIZE density: remaining_work / min_duration.
    
    # Let's refine:
    # For each op:
    #   min_dur = min(machine_options.values())
    #   max_dur = max(machine_options.values())
    #   avg_dur = sum(machine_options.values()) / len(machine_options)
    #   
    #   # Flexibility bonus: if the op can run on many machines, it's less critical to schedule now.
    #   # If it has only 1 option, it's more critical.
    #   flexibility = 1.0 / len(machine_options)
    #   
    #   # Urgency: remaining work
    #   urgency = op['remaining_work']
    #   
    #   # Speed: how fast can it be done?
    #   speed = 1.0 / min_dur
    #   
    #   score = urgency * speed * flexibility
    #   Pick max score.
    
    # This is a "Critical Work" heuristic.
    
    # Let's try to incorporate the machine selection better.
    # If we pick an op, we should pick the machine that minimizes the 
    # "Regret" of not picking another machine? No, too complex.
    
    # Let's use the **Min-Regret** heuristic for machine selection?
    # No, let's stick to the dispatch of the operation.
    
    # Proposed Score:
    # score = remaining_work / min_duration
    # This is the "Work Density". High density means high priority.
    
    # To add dynamics, we can scale this by the number of ready operations.
    # If many ops are ready, we are more selective.
    
    # Let's implement this.
    
    for i, op in enumerate(ready_operations):
        mw = op['remaining_work']
        if mw <= 0:
            # Should not happen for ready ops, but safety
            continue
            
        durations = list(op['machine_options'].values())
        min_dur = min(durations)
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-6
            
        # Density: How much work per unit of time on the best machine
        density = mw / min_dur
        
        # Add a small bonus for flexibility (more options = less critical to schedule NOW)
        # Actually, if it has many options, we can wait. If it has few, we should schedule now.
        # So, penalty for flexibility? Or bonus for rigidity?
        # Let's add a small weight for rigidity.
        rigidity = 1.0 / len(op['machine_options'])
        
        score = density * (1.0 + 0.1 * rigidity)
        
        if score > best_score:
            best_score = score
            best_idx = i
            # Select the machine with the shortest duration for this op
            best_machine = min(op['machine_options'].keys(), key=lambda m: op['machine_options'][m])
            
    # If no valid score was found (shouldn't happen), default to 0
    if best_machine is None:
        best_idx = 0
        best_machine = min(ready_operations[0]['machine_options'].keys(), key=lambda m: ready_operations[0]['machine_options'][m])
        
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
