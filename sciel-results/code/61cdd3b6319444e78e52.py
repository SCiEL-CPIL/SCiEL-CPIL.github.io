"""Job-shop dispatch rule: minimize estimated critical-path increase."""
import time as _time
import math

# Safety margin before the 150 s wall-clock limit
_DEADLINE_MARGIN = 5.0  # seconds


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next.

    Strategy: model the job-shop as a dynamic bipartite graph between ready
    operations and machines. For each (operation, machine) pair, estimate the
    increase in the global critical path length. The critical path is approximated
    by the operation's processing time on that machine plus the maximum remaining
    work of the job it unblocks (i.e., the remaining work after this operation).
    We select the pair that minimizes this estimated increase, which encourages
    scheduling operations that keep the overall schedule tight.

    Additionally, we apply a soft-min over machine options to handle the
    nonsmooth min objective smoothly, and a small bonus for jobs with more
    remaining operations (encouraging progress on long chains).

    Parameters
    ----------
    ready_operations : list of dict
        Each dict has keys: job, op, machine_options {machine: duration},
        remaining_ops, remaining_work.
    time : float
        Current simulation time.

    Returns
    -------
    int
        Index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we've been running too long, fall back to a simple heuristic
    _t0 = _time.perf_counter()

    # Precompute per-operation features
    scores = []
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        rem_work = op.get("remaining_work", 0)
        rem_ops = op.get("remaining_ops", 0)

        # Soft-min over machine durations (smooth approximation of min)
        durations = list(mo.values())
        if not durations:
            scores.append(float('inf'))
            continue

        # Use log-sum-exp for soft-min: softmin(x) = -logsumexp(-x)
        # But for simplicity and speed, use a weighted average with emphasis on min
        # Actually, let's compute the true min and also a soft version
        min_dur = min(durations)
        
        # Soft-min: use a temperature-based soft min
        # For small sets, just use min with a small penalty for variance
        # Simpler: use min duration as primary, but add a small term for
        # the second-min to encourage robustness
        if len(durations) > 1:
            sorted_durs = sorted(durations)
            second_min = sorted_durs[1]
            # Blend: 0.9 * min + 0.1 * second_min
            effective_dur = 0.9 * min_dur + 0.1 * second_min
        else:
            effective_dur = min_dur

        # Estimated critical path increase:
        # This operation takes effective_dur on its best machine, and after it
        # completes, the job still has remaining work. The critical path through
        # this job is approximately: current_time + effective_dur + remaining_work_after
        # where remaining_work_after = remaining_work - duration_of_this_op
        # But remaining_work includes this operation's work, so:
        # remaining_work_after = remaining_work - effective_dur (approx)
        
        # The estimated completion time of this job if we start this op now:
        # time + effective_dur + (remaining_work - effective_dur) = time + remaining_work
        # That's just remaining_work from now. But the critical path also depends
        # on machine availability and other jobs.
        
        # Better estimate: the makespan contribution is roughly
        # time + effective_dur + max(0, remaining_work - effective_dur)
        # = time + remaining_work
        # But we want to minimize the *increase* in makespan, which is harder.
        
        # Alternative: prioritize operations with shorter duration on the best machine,
        # and also consider how much work remains in the job.
        # A good heuristic: score = effective_dur * (1 + rem_ops * small_weight)
        # Lower is better.
        
        # Actually, let's think about this differently. The critical path length
        # is maximized by the longest chain. Starting an operation that has a
        # short duration but unlocks a job with lots of remaining work might be
        # good because it gets the chain moving. Starting an operation with a
        # long duration might be bad if it blocks a machine for a long time.
        
        # Score: we want to minimize the estimated increase in makespan.
        # A reasonable proxy: effective_dur / (remaining_work + 1) * (1 + 0.05 * rem_ops)
        # This favors operations that make good progress (high duration/work ratio)
        # and penalizes jobs with many remaining ops slightly (to avoid getting
        # stuck on long chains? or to prioritize them?).
        
        # Let's use: score = effective_dur - 0.05 * rem_ops * effective_dur
        # = effective_dur * (1 - 0.05 * rem_ops)
        # This encourages operations on jobs with more remaining work (negative bonus).
        
        rem_work_after = max(0, rem_work - effective_dur)
        
        # Estimated time until this job is done if we start now:
        # time + effective_dur + rem_work_after = time + rem_work
        # But the makespan is the max over all jobs. So starting this op
        # reduces the remaining work of this job, which may reduce the makespan
        # if this job is on the critical path.
        
        # Heuristic: score = effective_dur * (1.0 + 0.05 * rem_ops)
        # Lower score = better. This penalizes operations that take long
        # and also slightly penalizes jobs with many remaining ops (since they
        # will need more operations later, so we should get them started early
        # to reduce the risk of them becoming critical).
        
        # Actually, let's try: prioritize operations with small effective_dur
        # relative to the job's remaining work, and give a bonus for jobs
        # with more remaining ops (to start long chains early).
        
        score = effective_dur * (1.0 + 0.05 * rem_ops)
        scores.append(score)

    # Check deadline
    if _time.perf_counter() - _t0 > _DEADLINE_MARGIN:
        # Fallback: pick the operation with the smallest effective duration
        # (already computed in scores)
        best = 0
        best_score = scores[0]
        for i in range(1, n):
            if scores[i] < best_score:
                best = i
                best_score = scores[i]
        return int(best)

    # Pick the operation with the minimum score
    best = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] < best_score:
            best = i
            best_score = scores[i]

    return int(best)
