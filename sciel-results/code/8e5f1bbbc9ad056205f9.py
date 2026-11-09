"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and which machine).
    
    Strategy:
    - LRPT-dominant heuristic: prioritize jobs with more remaining work
    - Penalize long processing times (prefer shorter ops to free machines)
    - Slight urgency for jobs with more remaining operations
    
    Returns: (index, machine) tuple to explicitly select the best machine,
             or just index if only one option.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            return (0, best_machine)
        return 0

    # Build candidate list with best machine per operation
    candidates = []
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        # Choose machine that minimizes duration; tie-break by machine id for determinism
        min_machine = None
        min_dur = float('inf')
        for m, d in machine_options.items():
            if d < min_dur or (d == min_dur and (min_machine is None or m < min_machine)):
                min_dur = d
                min_machine = m

        if min_machine is None:
            continue

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)

        candidates.append({
            'index': i,
            'machine': min_machine,
            'min_dur': min_dur,
            'remaining_work': remaining_work,
            'remaining_ops': remaining_ops,
        })

    if not candidates:
        return 0

    if len(candidates) == 1:
        c = candidates[0]
        return (c['index'], c['machine'])

    # Compute normalization bounds
    max_rw = max(c['remaining_work'] for c in candidates)
    min_d = min(c['min_dur'] for c in candidates)
    max_d = max(c['min_dur'] for c in candidates)
    max_ro = max(c['remaining_ops'] for c in candidates)

    if max_rw <= 0:
        max_rw = 1.0
    if max_d == min_d:
        d_range = 1.0
    else:
        d_range = max_d - min_d
    if max_ro <= 0:
        max_ro = 1.0

    best_score = -float('inf')
    best_candidate = candidates[0]

    for c in candidates:
        # Remaining work: higher is better (critical path / LRPT)
        rw_norm = c['remaining_work'] / max_rw

        # Duration: shorter is better
        if d_range > 0:
            dur_norm = (c['min_dur'] - min_d) / d_range
        else:
            dur_norm = 0.0
        dur_score = 1.0 - dur_norm

        # Remaining ops: more remaining ops = more urgent
        ro_norm = c['remaining_ops'] / max_ro

        # Composite: LRPT dominant, then efficiency, then urgency
        score = 0.55 * rw_norm + 0.30 * dur_score + 0.15 * ro_norm

        if score > best_score:
            best_score = score
            best_candidate = c

    return (best_candidate['index'], best_candidate['machine'])
# EVOLVE-BLOCK-END
