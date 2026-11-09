"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We track per-call elapsed time and fall back to a fast heuristic
# if a single dispatch call is taking too long (shouldn't happen, but safe).
_CALL_DEADLINE = 0.5  # seconds per dispatch call; very generous


def _remaining_critical_path(op):
    """
    Estimate the remaining critical path length for a job given its current
    operation. The critical path is the sum of minimum processing times for
    all remaining operations in the job (including the current one).
    
    We use the minimum duration across machine options for each remaining op
    as a lower bound on how long that op will take, giving a precedence-aware
    urgency measure.
    """
    remaining_ops = op.get("remaining_ops", [])
    if not remaining_ops:
        # Use the current op's min duration as fallback
        mo = op.get("machine_options", {})
        if mo:
            return min(mo.values())
        return 0
    
    # remaining_ops should contain the current op and all subsequent ops
    # Each entry is typically a dict with 'machine_options'
    total = 0.0
    for r_op in remaining_ops:
        mo = r_op.get("machine_options", {})
        if mo:
            total += min(mo.values())
        else:
            total += 0.0
    return total


def _machine_contention_score(op, ready_operations, time):
    """
    Estimate how contested the best machine for this operation is.
    If many ready operations want the same machine, that machine is congested.
    We return a penalty: higher means more contention.
    """
    mo = op.get("machine_options", {})
    if not mo:
        return 0.0
    
    # Find the machine with the shortest duration for this op
    best_machine = min(mo, key=lambda m: mo[m])
    
    # Count how many other ready ops also want this machine (as one of their options)
    contention = 0
    for other in ready_operations:
        if other is op:
            continue
        other_mo = other.get("machine_options", {})
        if best_machine in other_mo:
            # Weight by how much the other op prefers this machine
            # (shorter duration relative to its options = stronger preference)
            other_min = min(other_mo.values())
            if other_mo[best_machine] == other_min:
                contention += 1
            else:
                contention += 0.3
    
    return contention


def dispatch(ready_operations, time):
    """
    Dispatch rule: select the ready operation with the highest priority score.
    
    Priority combines:
    1. Remaining critical path length (precedence-aware urgency)
    2. Remaining work (total workload remaining in the job)
    3. Machine contention avoidance (prefer ops that can use less-contested machines)
    4. Shortest processing time on the best machine (tie-breaker / efficiency)
    
    Returns: index of the ready operation to start, or (index, machine) tuple.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo, key=lambda m: mo[m])
            return (0, best_m)
        return 0
    
    # Deadline guard
    t_start = _time.monotonic()
    
    # Compute scores for each ready operation
    scores = []
    for i, op in enumerate(ready_operations):
        # 1. Remaining critical path (primary signal)
        rcp = _remaining_critical_path(op)
        
        # 2. Remaining work (secondary signal, captures total load)
        rw = op.get("remaining_work", 0.0)
        
        # 3. Machine contention (penalty)
        contention = _machine_contention_score(op, ready_operations, time)
        
        # 4. Best machine duration (shorter is better -> subtract)
        mo = op.get("machine_options", {})
        if mo:
            best_dur = min(mo.values())
        else:
            best_dur = 0.0
        
        # Composite score: high RCP and high RW are good; high contention and long best_dur are bad
        # Normalize-ish: RCP and RW are in similar units (time), contention is count, best_dur is time
        score = 1.0 * rcp + 0.5 * rw - 0.3 * contention - 0.2 * best_dur
        
        scores.append(score)
    
    # Check deadline
    if _time.monotonic() - t_start > _CALL_DEADLINE:
        # Fallback: just pick max remaining work
        return int(np.argmax([op.get("remaining_work", 0) for op in ready_operations]))
    
    best_idx = int(np.argmax(scores))
    
    # Optionally return the best machine for the chosen operation
    op = ready_operations[best_idx]
    mo = op.get("machine_options", {})
    if mo:
        # Pick the machine with the shortest duration for this op
        best_m = min(mo, key=lambda m: mo[m])
        return (best_idx, best_m)
    
    return best_idx
# EVOLVE-BLOCK-END
