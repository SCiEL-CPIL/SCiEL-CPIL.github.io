"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0


def _check_deadline():
    return _time.monotonic() >= _DEADLINE


def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute features for each ready operation
    remaining_works = np.empty(n, dtype=np.float64)
    remaining_ops = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    mean_durations = np.empty(n, dtype=np.float64)
    job_ids = np.empty(n, dtype=np.int64)

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        ro = op["remaining_ops"]
        durations = list(op["machine_options"].values())
        remaining_works[i] = rw
        remaining_ops[i] = ro
        min_durations[i] = min(durations) if durations else 0.0
        mean_durations[i] = np.mean(durations) if durations else 0.0
        job_ids[i] = op["job"]

    # Normalize each feature to [0, 1] range (higher is better for scoring)
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    md_min, md_max = min_durations.min(), min_durations.max()
    mean_d_min, mean_d_max = mean_durations.min(), mean_durations.max()

    # MWR: more remaining work = higher priority (urgency)
    if rw_max > rw_min:
        mwr_score = (remaining_works - rw_min) / (rw_max - rw_min)
    else:
        mwr_score = np.ones(n, dtype=np.float64)

    # Fewer remaining ops = closer to completion = higher priority
    if ro_max > ro_min:
        few_ops_score = 1.0 - (remaining_ops - ro_min) / (ro_max - ro_min)
    else:
        few_ops_score = np.ones(n, dtype=np.float64)

    # SPT: shorter min duration = less machine blocking = higher priority
    if md_max > md_min:
        spt_score = 1.0 - (min_durations - md_min) / (md_max - md_min)
    else:
        spt_score = np.ones(n, dtype=np.float64)

    # Mean duration based score
    if mean_d_max > mean_d_min:
        mean_d_score = 1.0 - (mean_durations - mean_d_min) / (mean_d_max - mean_d_min)
    else:
        mean_d_score = np.ones(n, dtype=np.float64)

    # Work per remaining op: higher means each op takes longer, more critical
    if ro_max > 0:
        work_per_op = remaining_works / np.maximum(remaining_ops, 1.0)
        wpo_min, wpo_max = work_per_op.min(), work_per_op.max()
        if wpo_max > wpo_min:
            wpo_score = (work_per_op - wpo_min) / (wpo_max - wpo_min)
        else:
            wpo_score = np.ones(n, dtype=np.float64)
    else:
        wpo_score = np.ones(n, dtype=np.float64)

    # Weighted combination with tuned weights
    # MWR dominates, SPT secondary, wpo tertiary, few_ops and mean_d supplementary
    weights = (0.35, 0.25, 0.20, 0.10, 0.10)
    combined = (weights[0] * mwr_score +
                weights[1] * spt_score +
                weights[2] * wpo_score +
                weights[3] * few_ops_score +
                weights[4] * mean_d_score)

    best_idx = int(np.argmax(combined))

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
