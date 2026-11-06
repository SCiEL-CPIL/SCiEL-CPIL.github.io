"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate metrics for each ready operation
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []

    for op in ready_operations:
        opts = op["machine_options"]
        # Find the fastest machine for this operation (SPT)
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # Load-Balancing Water-Filling heuristic with smooth surrogate:
    # For each candidate operation, simulate assigning it to the least-loaded
    # compatible machine and score by the resulting increase in the smooth
    # max (log-sum-exp) of machine loads. This captures the overall load
    # distribution better than the hard max.
    
    # Collect all machines that appear in any op's machine_options
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())
    
    # Estimate machine loads: for each machine, sum the durations of ops
    # that can use it, with higher weight for ops where this machine is optimal
    machine_loads = {m: 0.0 for m in all_machines}
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_m = best_machines[i]
        for m, dur in opts.items():
            # Weight: 1.0 if this is the best machine for this op, 0.5 otherwise
            weight = 1.0 if m == best_m else 0.5
            machine_loads[m] += dur * weight
    
    # Smooth max (log-sum-exp) surrogate for the L-infinity objective
    # This is differentiable and captures the "tail" of the load distribution
    def smooth_max(loads, temperature=1.0):
        if not loads:
            return 0.0
        arr = np.array(loads, dtype=np.float64)
        if len(arr) == 1:
            return float(arr[0])
        # Subtract max for numerical stability
        shifted = arr - np.max(arr)
        # Use log-sum-exp: max + log(sum(exp(shifted)))
        # For large values, this approximates max; for small, it's smoother
        log_sum = np.log(np.sum(np.exp(shifted / temperature)))
        return float(np.max(arr) + log_sum * temperature)
    
    # Compute current smooth max load
    load_values = list(machine_loads.values())
    if load_values:
        current_smooth_max = smooth_max(load_values, temperature=1.0)
        current_mean_load = sum(load_values) / len(load_values)
    else:
        current_smooth_max = 0.0
        current_mean_load = 0.0
    
    # Score each candidate operation:
    # Simulate assigning it to the least-loaded compatible machine
    # and compute the resulting increase in smooth max machine load
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]
        
        # Find the least-loaded compatible machine
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        
        # Simulate: add this op's duration to that machine's load
        new_load = machine_loads.get(min_load_machine, 0.0) + opts[min_load_machine]
        
        # Compute new smooth max load with the updated machine load
        new_loads = list(machine_loads.values())
        # Find the index of min_load_machine in the loads list
        machine_list = list(machine_loads.keys())
        m_idx = machine_list.index(min_load_machine)
        new_loads[m_idx] = new_load
        
        new_smooth_max = smooth_max(new_loads, temperature=1.0)
        
        # Score: increase in smooth max load (lower is better)
        load_increase = new_smooth_max - current_smooth_max
        
        # Secondary: how much does this op's machine deviate from mean after assignment?
        deviation = abs(new_load - current_mean_load)
        
        # Tertiary: prefer operations with fewer remaining ops (finish jobs sooner)
        # and shorter processing times (SPT)
        score = (
            load_increase,
            deviation,
            0 if is_last_ops[i] else 1,  # Last ops first
            min_durs[i],                  # SPT
            -remaining_works[i]           # LPT tie-break
        )
        scores.append(score)
    
    # Pick the operation with the best (lowest) score
    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
