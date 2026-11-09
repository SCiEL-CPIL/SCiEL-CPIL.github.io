"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _clock
import math

_T0 = _clock.time()
_LIMIT = 140.0


def _elapsed():
    return _clock.time() - _T0


def dispatch(ready_operations, time):
    if _elapsed() > _LIMIT:
        return 0
    n_ops = len(ready_operations)
    if n_ops <= 1:
        return 0

    def _priority(op, idx):
        opts = op["machine_options"]
        rem = op["remaining_work"]
        rem_ops = op.get("remaining_ops", 0)
        n_machines = len(opts)
        
        if n_machines == 0:
            return (-1e10, -idx)
        
        durations = list(opts.values())
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / n_machines
        
        # Constrained operations (fewer machine options) are harder to reschedule
        # Prioritize them to avoid getting stuck
        constraint_bonus = 1.0 / (1.0 + n_machines)
        
        # Urgency: more remaining work means this job is further from completion
        # But we also want to consider that operations with more remaining ops
        # in their job should be done earlier to unblock subsequent ops
        urgency = rem + 0.5 * rem_ops * min_dur
        
        # Prefer shorter minimum duration to free up machines quickly
        # But penalize if the max duration is much larger (unpredictable)
        duration_score = -min_dur - 0.3 * (max_dur - min_dur)
        
        # Combine: urgency dominates, then duration efficiency, then constraint
        priority = urgency + duration_score + 0.2 * constraint_bonus * rem
        
        return (priority, -idx)

    ranked = [(_priority(op, i), i) for i, op in enumerate(ready_operations)]
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]
# EVOLVE-BLOCK-END
