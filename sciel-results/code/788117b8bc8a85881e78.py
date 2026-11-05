"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Look-ahead dispatch rule for job-shop scheduling.

    Simulates the immediate completion of each ready operation and scores it
    by the reduction in the global critical path (max remaining work over jobs).
    This is a dynamic, state-based evaluation that directly targets the
    makespan lower bound, refined with a small smooth tie-break (SPT, urgency).
    """
    if not ready_operations:
        return 0

    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Precompute per-operation features
    min_durations = np.array(
        [min(op["machine_options"].values()) if op["machine_options"] else 1.0
         for op in ready_operations], dtype=np.float64)
    remaining_works = np.array(
        [op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array(
        [op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Job-level remaining work = max over that job's ready operations
    job_remaining_work = {}
    for op in ready_operations:
        job = op["job"]
        rw = op["remaining_work"]
        if job not in job_remaining_work or rw > job_remaining_work[job]:
            job_remaining_work[job] = rw

    jobs = list(job_remaining_work.keys())
    job_idx = {j: i for i, j in enumerate(jobs)}
    n_jobs = len(jobs)
    job_work = np.array([job_remaining_work[j] for j in jobs], dtype=np.float64)
    current_cp = job_work.max() if n_jobs else 0.0

    # Per-operation job index and its remaining work
    op_job_idx = np.array([job_idx[op["job"]] for op in ready_operations], dtype=np.int64)

    # ---- Look-ahead: reduction in global critical path ----
    # For each ready op i:
    #   new work of its job = remaining_work[i] - min_duration[i]
    #   new global CP = max( new_work_of_job_i, max of other jobs' work )
    #   reduction = current_cp - new_global_cp
    # We want the op that yields the greatest decrease.

    # max of each job's work excluding itself (for the "other jobs" term)
    # Build per-job max over ready ops; that's job_work. The "others" max for
    # job j is the max of job_work over all jobs except j.
    # Compute via prefix/suffix max over the job array.
    prefix_max = np.zeros(n_jobs, dtype=np.float64)
    suffix_max = np.zeros(n_jobs, dtype=np.float64)
    if n_jobs > 0:
        prefix_max[0] = job_work[0]
        for j in range(1, n_jobs):
            prefix_max[j] = max(prefix_max[j - 1], job_work[j])
        suffix_max[n_jobs - 1] = job_work[n_jobs - 1]
        for j in range(n_jobs - 2, -1, -1):
            suffix_max[j] = max(suffix_max[j + 1], job_work[j])

    others_max = np.zeros(n_jobs, dtype=np.float64)
    for j in range(n_jobs):
        left = prefix_max[j - 1] if j > 0 else -np.inf
        right = suffix_max[j + 1] if j < n_jobs - 1 else -np.inf
        others_max[j] = max(left, right)

    # For each ready op, compute the look-ahead reduction
    reductions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        j = op_job_idx[i]
        new_job_work = remaining_works[i] - min_durations[i]
        new_global_cp = max(new_job_work, others_max[j])
        reductions[i] = current_cp - new_global_cp

    # Normalize reductions to [0,1]
    r_min = reductions.min()
    r_max = reductions.max()
    if r_max - r_min < 1e-12:
        reductions_norm = np.zeros(n, dtype=np.float64)
    else:
        reductions_norm = (reductions - r_min) / (r_max - r_min)

    # Smooth tie-break terms (small weight), also normalized
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    spt_norm = 1.0 - normalize(min_durations)
    rem_ops_norm = 1.0 - normalize(remaining_ops)

    # Composite: look-ahead reduction dominates; SPT and urgency refine ties.
    final_scores = (
        reductions_norm * 0.80 +
        spt_norm * 0.12 +
        rem_ops_norm * 0.08
    )

    # Tie-breaking: among near-max scores, prefer shorter min_duration
    max_score = final_scores.max()
    close_to_max = np.where(final_scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        tied_durations = min_durations[close_to_max]
        best_tie_idx = close_to_max[np.argmin(tied_durations)]
        return int(best_tie_idx)

    return int(np.argmax(final_scores))
# EVOLVE-BLOCK-END
