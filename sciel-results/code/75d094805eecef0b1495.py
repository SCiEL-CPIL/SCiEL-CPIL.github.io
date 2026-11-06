"""Dispatch rule for job-shop scheduling with look-ahead critical path reduction."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 110.0  # safety margin before 120s kill
_start_time = None


def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.monotonic()

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _time.monotonic() - _start_time > _DEADLINE:
        # Fallback: max remaining work
        return int(max(range(n), key=lambda i: ready_operations[i]["remaining_work"]))

    # Current global critical path: max remaining work across all jobs
    # We approximate this using the ready_operations' remaining_work values
    # But we also need to consider that completing an operation reduces its job's remaining work
    # and potentially unlocks the next operation (which adds to the critical path if it's on a bottleneck machine)

    # Strategy: For each ready operation, simulate its completion:
    # 1. Its job's remaining_work decreases by the duration of this op (on chosen machine)
    # 2. The next op in that job becomes ready (if any), adding its min duration to that job's remaining work
    # 3. Compute new max remaining work across all jobs
    # Pick the op that minimizes the new max (i.e., greatest reduction in makespan lower bound)

    # First, gather job -> remaining_work mapping from ready ops
    # Note: multiple ops from same job may be ready; we track per-job
    job_remaining = {}
    for i, op in enumerate(ready_operations):
        j = op["job"]
        if j not in job_remaining:
            job_remaining[j] = op["remaining_work"]
        else:
            # If multiple ops from same job are ready, remaining_work should be consistent
            # or we take the max (it represents total remaining work for the job)
            job_remaining[j] = max(job_remaining[j], op["remaining_work"])

    # Current critical path
    current_cp = max(job_remaining.values()) if job_remaining else 0

    best_idx = 0
    best_new_cp = float('inf')

    for i, op in enumerate(ready_operations):
        j = op["job"]
        machine_options = op["machine_options"]

        # Choose the machine with minimum duration for this op (greedy machine selection)
        # This is a heuristic: pick the fastest machine
        min_dur = min(machine_options.values())
        min_machine = min(machine_options, key=machine_options.get)

        # Simulate: this op completes
        # Job j's remaining work reduces by min_dur
        old_rem = job_remaining.get(j, 0)
        new_rem_j = old_rem - min_dur

        # The next op in job j (if any) becomes ready
        # remaining_ops tells us how many ops are left after this one
        # If remaining_ops > 0, the next op exists. Its duration is unknown exactly,
        # but we can estimate: the remaining_work already includes it.
        # Actually, remaining_work = sum of durations of all remaining ops (including this one? or after?)
        # Let's assume remaining_work includes the current op's duration.
        # After completing this op, remaining work for job j = old_rem - min_dur
        # The next op's duration is part of that remaining work, so no additional addition needed.
        # The critical path for job j becomes new_rem_j.

        # But wait: if remaining_work does NOT include the current op, then:
        # new_rem_j = old_rem (the remaining ops after this one)
        # Let's handle both cases conservatively.
        # Given the interface says "remaining_work", it likely means work remaining AFTER this op,
        # or including this op. Let's assume it includes this op (more common in such interfaces).
        # So after completion: job j has remaining_work - min_dur left.

        # Compute new critical path
        # We need to consider all jobs, not just those with ready ops.
        # But we only have info about jobs in ready_operations.
        # For jobs not in ready_operations, their remaining work is unknown but >= 0.
        # We'll only consider jobs we have data for.

        new_job_remaining = dict(job_remaining)
        new_job_remaining[j] = new_rem_j

        new_cp = max(new_job_remaining.values()) if new_job_remaining else 0

        # Tie-breaker: prefer ops on machines with less contention (fewer other ready ops on same machine)
        # Count how many other ready ops are on min_machine
        machine_contention = sum(
            1 for k, op2 in enumerate(ready_operations)
            if k != i and min_machine in op2["machine_options"]
        )

        # Score: primary is new_cp (lower is better), secondary is machine_contention (lower is better)
        # We want to minimize new_cp, then minimize contention
        if new_cp < best_new_cp - 1e-9:
            best_new_cp = new_cp
            best_idx = i
        elif abs(new_cp - best_new_cp) < 1e-9:
            # Tie-break: prefer lower contention, then higher remaining work (urgency)
            old_contention = sum(
                1 for k, op2 in enumerate(ready_operations)
                if k != best_idx and min(ready_operations[best_idx]["machine_options"], key=ready_operations[best_idx]["machine_options"].get) in op2["machine_options"]
            )
            if machine_contention < old_contention:
                best_idx = i
            elif machine_contention == old_contention:
                # Prefer op with more remaining work (more urgent)
                if op["remaining_work"] > ready_operations[best_idx]["remaining_work"]:
                    best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
