"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.

    Strategy: balance job-level urgency against the immediate machine impact.
      - Job urgency  : remaining work spread over remaining ops (criticality).
      - Machine cost : shortest processing time among the op's machine options,
                       which reduces the delay this op imposes on a machine.

    A weighted blend (urgency dominates, SPT breaks ties / refines) is used.
    """
    if _deadline_exceeded():
        # Fallback: most remaining work (cheap, safe).
        return int(max(range(len(ready_operations)),
                       key=lambda i: ready_operations[i]["remaining_work"]))

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    best_idx = 0
    best_score = float("-inf")

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            # Shortest processing time the op can take (min over machines).
            spt = min(machine_options.values())
            if spt <= 0:
                spt = 1.0
        else:
            spt = 1.0

        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1

        # Job urgency: average remaining work per remaining op.
        urgency = remaining_work / remaining_ops

        # Blend: urgency (weight 1.0) plus a small SPT term (lower is better).
        score = urgency - 0.15 * spt

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
