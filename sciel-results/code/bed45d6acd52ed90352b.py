"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Track start time for deadline guard
_t0 = _time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule using a Lexicographic Critical Path Strategy.
    
    Structural Change: Instead of a weighted sum heuristic, this rule uses a
    strict lexicographic ordering based on:
    1. Remaining Work (Descending): Prioritizes jobs with the longest remaining
       critical path (approximated by total remaining work).
    2. Remaining Operations (Descending): Ties broken by jobs with more steps
       remaining, as they have more opportunity to cause delays.
    3. Minimum Duration (Ascending): Ties broken by the operation that can be
       completed fastest on its best available machine.
    
    This ensures the most critical job (longest path) is never delayed by less
    critical jobs, and among critical jobs, we pick the one that frees up the
    machine/job state fastest.
    """
    # Deadline guard
    if _time.time() - _t0 > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Prepare a list of tuples for sorting: (neg_remaining_work, neg_remaining_ops, min_duration, index, best_machine)
    # We use negative values for descending order on work and ops.
    candidates = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Find the shortest processing time and the corresponding machine
        min_duration = float('inf')
        best_machine = None
        
        for machine, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                best_machine = machine
        
        # If no machines found (should not happen), default to 0 duration and None machine
        if best_machine is None:
            min_duration = 0
            best_machine = 0
            
        candidates.append((
            -remaining_work,       # 1. Max Remaining Work (Critical Path)
            -remaining_ops,        # 2. Max Remaining Ops
            min_duration,          # 3. Min Duration (Fastest completion)
            i,                     # Index
            best_machine           # Selected Machine
        ))
    
    # Sort the candidates
    # This is a structural change: deterministic lexicographic sort vs weighted sum
    candidates.sort()
    
    # The first candidate is the best
    best_candidate = candidates[0]
    best_idx = best_candidate[3]
    best_machine = best_candidate[4]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
