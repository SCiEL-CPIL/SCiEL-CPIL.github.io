"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # seconds wall-clock guard (evaluator kills at 150s)
_START_TIME = None


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Returns:
        int: index of the ready operation to start
        or (int, machine): index and chosen machine
    
    Strategy:
    1. Prioritize operations with fewest remaining machine options (constraint tightness)
       to prevent future starvation.
    2. Break ties by selecting the operation with the shortest minimum processing time
       on any eligible machine (to minimize immediate bottleneck impact).
    3. For machine selection, pick the machine that minimizes projected idle time
       relative to the operation's processing time, with a slight preference for
       machines with more total remaining work (load balancing).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        op = ready_operations[0]
        machines = op["machine_options"]
        if machines:
            # Pick machine with minimum duration
            best_m = min(machines, key=lambda m: machines[m])
            return (0, best_m)
        return 0
    
    # Compute scores for each ready operation
    # Score = (num_machine_options, min_duration, -remaining_work)
    # Lower is better: fewer options first, then shorter min duration, then more work
    
    scores = []
    for i, op in enumerate(ready_operations):
        machines = op["machine_options"]
        if not machines:
            scores.append((999999, 999999, 0))
            continue
        
        num_opts = len(machines)
        min_dur = min(machines.values())
        rem_work = op.get("remaining_work", 0)
        
        # Constraint tightness: fewer options = higher priority
        # Then shorter min processing time = higher priority (less blocking)
        # Then more remaining work = higher priority (longer jobs should start earlier)
        scores.append((num_opts, min_dur, -rem_work))
    
    # Find the operation with the best (lowest) score
    best_idx = int(np.argmin([s[0] * 1000000 + s[1] * 1000 - s[2] for s in scores]))
    
    # Now select the best machine for this operation
    op = ready_operations[best_idx]
    machines = op["machine_options"]
    
    if not machines:
        return best_idx
    
    if len(machines) == 1:
        m = list(machines.keys())[0]
        return (best_idx, m)
    
    # Machine selection heuristic:
    # For each machine, estimate the "cost" of assigning this operation to it.
    # We want to minimize the makespan contribution.
    # Heuristic: prefer machines where this operation's duration is a small fraction
    # of the machine's total remaining work (load balancing), and prefer shorter
    # processing times.
    
    # Since we don't have direct access to machine load, we use a simpler heuristic:
    # Pick the machine with the shortest processing time for this operation.
    # If tied, pick the one with the lower machine id (deterministic).
    
    best_machine = min(machines.items(), key=lambda x: (x[1], x[0]))[0]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
