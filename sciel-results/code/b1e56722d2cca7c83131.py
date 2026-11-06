"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    _start = _time.perf_counter()
    _deadline = _start + 115.0  # 115s safety margin (evaluator kills at 150s)

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Deadline guard: if we're running out of time, return best so far
        if _time.perf_counter() > _deadline:
            break

        remaining_work = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})

        if not machine_options:
            score = 0.0
            machine = None
        else:
            # Find the minimum duration among eligible machines (SPT on machine choice)
            min_duration = min(machine_options.values())
            best_machine_for_op = min(machine_options, key=machine_options.get)

            if min_duration <= 0:
                min_duration = 1e-9
                best_machine_for_op = list(machine_options.keys())[0]

            # Composite heuristic:
            # 1) Urgency: higher remaining_work -> more urgent
            # 2) Speed: shorter min_duration -> prefer (SPT-like)
            # 3) Depth: fewer remaining_ops -> closer to completion -> prefer
            #
            # We use a multiplicative combination that rewards both
            # high remaining work AND short processing time,
            # with a small bonus for operations closer to finishing.
            #
            # score = (remaining_work + 1.0) / min_duration  +  1.0 / (remaining_ops + 1)
            #
            # The "+1.0" in the numerator acts as a pure SPT tie-breaker.
            # The depth term gently favors operations that are near completion
            # to reduce tail latency.

            work_term = (remaining_work + 1.0) / min_duration
            depth_term = 1.0 / (remaining_ops + 1)
            score = work_term + 0.3 * depth_term
            machine = best_machine_for_op

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = machine

    # Return index only (matching original interface)
    return int(best_index)
# EVOLVE-BLOCK-END
