import numpy as np
import time as time_module

_START_TIME = time_module.time()
_DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def _elapsed_ok():
    return (time_module.time() - _START_TIME) < _DEADLINE_SECONDS

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    if not _elapsed_ok():
        return (0, None)

    best_index = 0
    best_machine = None
    best_score = -float('inf')

    # Precompute max remaining_work for normalization
    max_rw = 0.0
    for op in ready_operations:
        rw = op.get("remaining_work", 0.0)
        if rw > max_rw:
            max_rw = rw
    if max_rw == 0.0:
        max_rw = 1.0

    for i, op in enumerate(ready_operations):
        if not _elapsed_ok():
            break

        machine_options = op["machine_options"]
        if not machine_options:
            continue

        # Use log-sum-exp soft-min to capture the full distribution
        durations = list(machine_options.values())
        # Soft-min: -log(sum(exp(-d/tau))) * tau
        # For small tau, approximates min; for larger tau, smoother
        tau = 1.0
        neg_d_over_tau = np.array([-d / tau for d in durations])
        max_nd = np.max(neg_d_over_tau)
        soft_min_val = -tau * (max_nd + np.log(np.sum(np.exp(neg_d_over_tau - max_nd))))

        # Find the machine with the shortest duration (for assignment)
        min_dur = float('inf')
        min_machine = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                min_machine = m

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)

        # Normalized remaining work
        norm_rw = remaining_work / max_rw

        # Composite score: emphasize remaining work, penalize processing time
        # Blend: primary signal is remaining work, secondary is soft-min duration
        # Use a weighted sum where remaining_work dominates but duration matters
        score = norm_rw * 10.0 - soft_min_val * 0.1 - remaining_ops * 0.01

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_machine

    return (best_index, best_machine)
