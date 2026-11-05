"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import math


def _soft_min(values, tau=1.0):
    """
    Smooth approximation of min using log-sum-exp.
    soft_min(x) = -log(sum(exp(-x_i / tau)) * tau)
    As tau -> 0, approaches min(values).
    For larger tau, gives a smoother, more robust measure.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Use log-sum-exp for numerical stability
    neg_scaled = np.array([-v / tau for v in values])
    max_neg = np.max(neg_scaled)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    soft_min_val = -tau * log_sum
    return soft_min_val


def dispatch(ready_operations, time):
    """
    Dispatch rule that balances:
    1. Urgency: jobs with more remaining work (critical path)
    2. Machine load balancing: prefer machines with less total assigned work
    3. Duration efficiency: prefer shorter processing times when possible
    4. Machine congestion: penalize operations that want the same machine as others
    
    Uses soft-min for robustness to ties and noise in duration options.
    
    Returns index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard: return a safe choice if we're running out of time
    start_time = _time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    # First pass: compute base scores and track machine usage
    machine_counts = {}  # machine -> count of ops wanting it
    machine_durations = {}  # machine -> list of durations for ops wanting it
    
    base_scores = np.zeros(n)
    best_machine = [None] * n
    soft_min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            base_scores[i] = -1.0
            continue
        
        durations = list(machine_options.values())
        machines = list(machine_options.keys())
        
        # Use soft-min for robustness (tau=0.5 for mild smoothing)
        sm_dur = _soft_min(durations, tau=0.5)
        soft_min_durations[i] = sm_dur
        
        # Find the machine with the shortest duration (tie-break by machine id for determinism)
        min_dur = min(durations)
        # Prefer the first machine with min duration
        best_m = None
        for m, d in machine_options.items():
            if d == min_dur:
                best_m = m
                break
        best_machine[i] = best_m
        
        # Track machine usage
        if best_m is not None:
            machine_counts[best_m] = machine_counts.get(best_m, 0) + 1
            if best_m not in machine_durations:
                machine_durations[best_m] = []
            machine_durations[best_m].append(min_dur)
        
        # Base score: urgency - efficiency
        alpha = 1.0
        beta = 0.1
        base_scores[i] = alpha * remaining_work - beta * sm_dur
    
    # Second pass: add congestion penalty
    # Penalize operations that share their preferred machine with others
    # The more ops wanting the same machine, the higher the congestion
    
    for i, op in enumerate(ready_operations):
        if not op.get("machine_options", {}):
            continue
        
        best_m = best_machine[i]
        if best_m is None:
            continue
        
        count = machine_counts.get(best_m, 1)
        
        # Congestion penalty: proportional to (count - 1) * average duration on that machine
        if count > 1:
            avg_dur_on_machine = sum(machine_durations.get(best_m, [soft_min_durations[i]])) / len(machine_durations.get(best_m, [1]))
            congestion_penalty = 0.05 * (count - 1) * avg_dur_on_machine
        else:
            congestion_penalty = 0.0
        
        scores[i] = base_scores[i] - congestion_penalty
    
    # Check deadline before final selection
    if _time.time() > deadline:
        # Return the argmax of base scores as fallback
        return int(np.argmax(base_scores))
    
    # Pick the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
