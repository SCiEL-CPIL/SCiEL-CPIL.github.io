"""Open-Shop Scheduling dispatch rule for open-shop event-driven simulation."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE_MARGIN = 5  # seconds before the 150s kill; be conservative

# Track machine loads to make machine-aware decisions
_machine_load = {}


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Structural change: Machine-load-aware dispatch.
    Instead of purely job-centric LRPT/LPT, this uses a combined score:
    - Job criticality: remaining_work (LRPT)
    - Machine congestion: current load on the best machine for this op
    - Operation urgency: ratio of duration to remaining work
    
    The score rewards starting operations on less-loaded machines when
    job criticality is similar, reducing idle time and balancing load.
    
    Return: (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = _best_machine_for_op(op)
        return (0, best_m)

    # Deadline guard: if we are too close to the limit, return a quick heuristic
    if _elapsed() > 145:
        op = ready_operations[0]
        best_m, _ = _best_machine_for_op(op)
        return (0, best_m)

    # Compute machine loads from current time perspective
    # We don't have direct access to future completions, but we can use
    # the current simulation time as a proxy for when machines become free.
    # A simpler approach: track cumulative assigned work per machine.
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        durations = op["machine_options"]
        if not durations:
            scores.append(-1)
            best_machines.append(None)
            continue
            
        # Find the machine with shortest duration (SPT) for this operation
        best_m, min_dur = _best_machine_for_op(op)
        
        # Get current load on that machine (use 0 if not tracked)
        machine_load = _machine_load.get(best_m, 0)
        
        # Compute score with three components:
        # 1. Job criticality: higher remaining_work = more urgent
        # 2. Machine availability: lower load on best machine = better
        # 3. Urgency ratio: duration / remaining_work (higher = more pressing)
        
        # Normalize remaining_work to [0,1] scale for fair weighting
        # Use a soft normalization: rw / (rw + 1) to avoid division issues
        criticality = rw / (rw + 1.0) if rw > 0 else 0.0
        
        # Machine congestion penalty: higher load = worse
        # Scale load relative to typical durations
        congestion_penalty = machine_load / (machine_load + min_dur + 1.0) if min_dur > 0 else 0.5
        
        # Urgency: operations with high duration relative to remaining work
        # are more critical to start now
        urgency = min_dur / (rw + min_dur) if (rw + min_dur) > 0 else 0.5
        
        # Combined score: weight criticality heavily, but let machine load
        # break ties and influence when criticality is similar
        score = (
            criticality * 100.0 +      # Primary: job criticality
            (1.0 - congestion_penalty) * 30.0 +  # Secondary: machine availability
            urgency * 10.0              # Tertiary: operation urgency
        )
        
        scores.append(score)
        best_machines.append(best_m)
    
    # Find index with max score
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    # Update machine load tracking
    if best_machine is not None:
        op = ready_operations[best_idx]
        dur = op["machine_options"].get(best_machine, 0)
        _machine_load[best_machine] = _machine_load.get(best_machine, 0) + dur
    
    if best_machine is None:
        return best_idx
    return (best_idx, best_machine)


def _best_machine_for_op(op):
    """
    For a given operation, select the machine that minimizes its duration.
    """
    durations = op["machine_options"]
    if not durations:
        return None, 0
    best_machine = min(durations, key=lambda m: durations[m])
    return best_machine, durations[best_machine]
# EVOLVE-BLOCK-END
