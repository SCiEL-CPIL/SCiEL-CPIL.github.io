"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=3.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation of min."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 1.0
    if len(values) == 1:
        return float(values[0])
    # Soft min: -1/alpha * log(sum(exp(-alpha * x)))
    neg_alpha_vals = -alpha * values
    max_neg = np.max(neg_alpha_vals)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_alpha_vals - max_neg)))
    return -log_sum / alpha

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, collect machine durations
    all_durations = []
    machine_sets = []
    for op in ready_operations:
        if op["machine_options"]:
            all_durations.append(list(op["machine_options"].values()))
            machine_sets.append(set(op["machine_options"].keys()))
        else:
            all_durations.append([1.0])
            machine_sets.append(set())

    # Compute duration statistics
    min_durations = np.array([min(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    max_durations = np.array([max(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    mean_durations = np.array([np.mean(d) if d else 1.0 for d in all_durations], dtype=np.float64)
    
    # Soft-min for smooth duration estimation
    soft_min_durations = np.array([
        _soft_min(np.array(d, dtype=np.float64), alpha=3.0) if d else 1.0 
        for d in all_durations
    ], dtype=np.float64)

    # WSPT ratio: remaining_work / soft_min_duration
    wspt = remaining_work / np.maximum(soft_min_durations, 1e-9)
    
    # Normalize all features
    n_wspt = _normalize(wspt)
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)
    nd_soft = _normalize(soft_min_durations)

    # Dynamic machine-load balancing strategy:
    # For each candidate operation i, estimate the "load" it would add to the least-loaded machine.
    # We approximate machine load by summing the minimum durations of all ready ops that can use that machine.
    # Then, for each candidate op, we assign it to the machine with the current lowest load.
    # The "resulting max load" is the maximum load across all machines after this assignment.
    # We want to minimize this resulting max load (balance).
    
    # Pre-compute initial machine loads (sum of min durations of ready ops available on each machine)
    machine_load = {}
    for i in range(n):
        for m in machine_sets[i]:
            if m not in machine_load:
                machine_load[m] = 0.0
            machine_load[m] += min_durations[i]
    
    # If no machines found (shouldn't happen), fallback
    if not machine_load:
        return int(np.argmax(nw))

    max_machine_load = max(machine_load.values()) if machine_load else 1.0
    
    # For each candidate op i, simulate assigning it to its least-loaded machine
    # and compute the resulting max machine load
    resulting_max_loads = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if not machine_sets[i]:
            resulting_max_loads[i] = max_machine_load
            continue
        
        # Find the machine with the minimum current load among available machines
        min_load_m = None
        min_load_val = float('inf')
        for m in machine_sets[i]:
            load = machine_load.get(m, 0.0)
            if load < min_load_val:
                min_load_val = load
                min_load_m = m
        
        if min_load_m is None:
            resulting_max_loads[i] = max_machine_load
            continue
        
        # The new load for this machine would be min_load_val + min_durations[i]
        new_load = min_load_val + min_durations[i]
        
        # The resulting max load is max(max_machine_load, new_load)
        # But note: max_machine_load might be from a different machine.
        # Since we are adding to the min-loaded machine, the new max is max(old_max, new_load)
        # unless the old max was from this same machine (which is impossible since it was the min).
        # Actually, if all machines have same load, it doesn't matter.
        # So resulting max load = max(max_machine_load, new_load)
        resulting_max_loads[i] = max(max_machine_load, new_load)
    
    # Normalize resulting max loads (lower is better for balance)
    n_rml = _normalize(resulting_max_loads)
    # We want to MINIMIZE n_rml, so we use (1 - n_rml) in the score to MAXIMIZE balance

    # Combine scores:
    # - n_wspt: higher WSPT is often better (urgency/efficiency)
    # - no: higher remaining ops (keep in motion)
    # - (1 - nd_soft): lower soft-min duration (SPT)
    # - (1 - n_rml): better load balance (lower resulting max load)
    
    w_wspt = 0.30
    w_ops = 0.25
    w_spt = 0.20
    w_balance = 0.25

    score = (w_wspt * n_wspt + 
             w_ops * no + 
             w_spt * (1.0 - nd_soft) + 
             w_balance * (1.0 - n_rml))

    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
