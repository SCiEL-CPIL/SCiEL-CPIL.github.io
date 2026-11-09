"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# We use a composite dispatch rule that combines:
# 1. Most Work Remaining First (MWRF) - proven baseline
# 2. Shortest Processing Time (SPT) on the selected machine
# 3. Explicit machine selection to reduce contention
#
# The key insight: when multiple ready operations can run on the same machine,
# we should prioritize the one with the longest remaining work (critical path),
# but when choosing which machine to use, pick the one with shortest duration.
#
# We also add a small tie-breaking heuristic: if two operations have similar
# remaining work, prefer the one that has fewer remaining ops (sooner completion).


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next, and optionally which machine.
    
    Returns:
        - int: index of the ready operation to start, OR
        - (int, machine): index and preferred machine
    
    Strategy:
        1. For each ready operation, compute a priority score.
        2. Priority = remaining_work / (1 + 0.1 * remaining_ops) 
           This favors jobs with more total work but penalizes those with many 
           remaining ops (they're less urgent in a different sense - actually 
           we want to prioritize jobs that are on the critical path).
        
        Let me reconsider: 
        - remaining_work: total processing time left for this job
        - remaining_ops: number of operations left for this job
        
        A job with high remaining_work is on the critical path.
        A job with few remaining_ops might be near completion.
        
        Better heuristic: prioritize by remaining_work (MWRF), 
        but break ties by choosing the operation that, when assigned to its 
        best (shortest) machine, has the shortest processing time (SPT).
        
        Actually, for makespan minimization, the standard approach is:
        - At each decision point, select the operation with the largest 
          remaining work (MWRF).
        - For machine selection, choose the machine that minimizes the 
          completion time of that operation (i.e., the machine with the 
          shortest duration for that operation, assuming it's free).
        
        Since we don't know machine availability in this interface, we'll 
        return the operation index and the machine with the shortest duration 
        for that operation.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute priority scores
    # Primary: remaining_work (higher is better)
    # Secondary: for tie-breaking, prefer operations where the shortest 
    #            machine duration is shorter (SPT-like)
    
    scores = []
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        
        # Find the minimum duration across available machines
        machine_options = op["machine_options"]
        if machine_options:
            min_dur = min(machine_options.values())
            # The machine with min duration
            best_machine = min(machine_options, key=machine_options.get)
        else:
            min_dur = 0
            best_machine = None
        
        # Composite score: 
        # We want high remaining_work, and for ties, lower min_dur is better
        # Use a weighted combination
        # Score = remaining_work * 1000 - min_dur (so higher is better)
        # But this might overweight remaining_work too much.
        
        # Alternative: normalize and combine
        # Let's use: score = remaining_work + alpha * (1 / (min_dur + 1))
        # where alpha is small, so remaining_work dominates
        
        # Actually, let's keep it simple: 
        # Primary sort key: remaining_work (descending)
        # Secondary sort key: min_dur (ascending) - SPT tie-break
        # Tertiary sort key: remaining_ops (ascending) - prefer jobs closer to done
        
        scores.append((rw, -min_dur, -ro, i))
    
    # Sort by score descending (we want highest rw, then lowest min_dur, then lowest ro)
    # Since we're taking argmax, we need to be careful with the sign
    # Let's just find the best index directly
    
    best_idx = 0
    best_score = None
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        machine_options = op["machine_options"]
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        
        # Composite score: higher is better
        # remaining_work is the dominant factor
        # -min_dur: lower duration is better (so we subtract)
        # -ro: fewer remaining ops is slightly better
        score = rw - 0.01 * min_dur - 0.001 * ro
        
        if best_score is None or score > best_score:
            best_score = score
            best_idx = i
    
    # For the chosen operation, find the best machine (shortest duration)
    chosen_op = ready_operations[best_idx]
    machine_options = chosen_op["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=machine_options.get)
        return (best_idx, best_machine)
    else:
        return best_idx


# EVOLVE-BLOCK-END
