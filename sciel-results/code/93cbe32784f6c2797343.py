import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0
_machine_available = {}
_machine_counts = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple urgency
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op["machine_options"]
        if mo:
            best_m, best_d = None, float('inf')
            for m, d in mo.items():
                avail = _machine_available.get(m, 0)
                count = _machine_counts.get(m, 0)
                effective_d = d + max(0, avail - time) * 0.5 + count * 0.01
                if effective_d < best_d:
                    best_d = effective_d
                    best_m = m
            _machine_counts[best_m] = _machine_counts.get(best_m, 0) + 1
            return (0, best_m)
        return 0

    # Calculate scores for each ready operation
    # Combine urgency (remaining work), efficiency (min duration), and machine availability
    scores = []
    for op in ready_operations:
        mo = op["machine_options"]
        if not mo:
            scores.append(0)
            continue

        min_d = min(mo.values())
        work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Urgency: total remaining work (higher = more urgent)
        urgency = work

        # Efficiency: how fast this op can be done (lower duration = better)
        efficiency = 1.0 / (min_d + 1e-6)

        # Critical path consideration: fewer remaining ops after this = more critical
        criticality = 1.0 / (remaining_ops + 1) if remaining_ops > 0 else 1.0

        # Machine availability: prefer machines that are free sooner
        min_avail = min(_machine_available.get(m, 0) for m in mo.keys())
        wait_penalty = max(0, min_avail - time)

        # Combined score
        score = urgency * 0.3 + efficiency * work * 0.5 + criticality * work * 0.2 - wait_penalty * 0.1
        scores.append(score)

    best_idx = int(np.argmax(scores))

    # Machine selection for best_idx
    op = ready_operations[best_idx]
    mo = op["machine_options"]
    if mo:
        best_m = None
        best_score_m = float('inf')
        for m, d in mo.items():
            avail = _machine_available.get(m, 0)
            count = _machine_counts.get(m, 0)
            wait = max(0, avail - time)
            # Prefer machines with shorter wait and lower load
            effective_score = d + wait * 0.5 + count * 0.05
            if effective_score < best_score_m:
                best_score_m = effective_score
                best_m = m

        _machine_counts[best_m] = _machine_counts.get(best_m, 0) + 1
        # Update available time for the selected machine
        d_selected = mo[best_m]
        avail_selected = _machine_available.get(best_m, 0)
        _machine_available[best_m] = max(avail_selected, time) + d_selected

        return (best_idx, best_m)

    return best_idx
