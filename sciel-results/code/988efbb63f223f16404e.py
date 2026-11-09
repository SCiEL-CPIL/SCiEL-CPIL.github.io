"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level deadline guard: we must finish within 120s per evaluation.
# The dispatch function is called many times per instance; we track total
# elapsed time across calls to ensure we don't exceed the budget.
_START_TIME = _time.monotonic()
TIME_BUDGET = 115.0  # safety margin before 120s limit


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Combine multiple heuristics into a composite score:
    1. Most work remaining (MWL) - prioritize jobs with more total work left
    2. Critical path proxy: remaining_work / remaining_ops - average op duration
    3. Machine load consideration: prefer operations that can go to less-loaded machines
    4. Soft-min over machine options to capture spread of available machines
    
    We use a weighted combination that balances urgency (remaining work) with
    machine availability (shorter processing times on available machines).
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Check time budget - if we're running low, fall back to fast MWL
    if _elapsed() > TIME_BUDGET - 5.0:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    scores = np.zeros(n)
    
    # Component 1: Most Work Remaining (normalized)
    rw = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    rw_max = rw.max()
    if rw_max > 0:
        scores += 1.0 * (rw / rw_max)
    
    # Component 2: Critical path proxy (average remaining op duration)
    ro = np.array([max(op["remaining_ops"], 1) for op in ready_operations], dtype=np.float64)
    avg_dur = rw / ro  # remaining_work / remaining_ops
    avg_max = avg_dur.max()
    if avg_max > 0:
        scores += 0.5 * (avg_dur / avg_max)
    
    # Component 3: Best machine option (shortest duration) - favors ops that
    # can be processed quickly
    best_durs = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            best_durs[i] = min(mo.values())
        else:
            best_durs[i] = 0.0
    
    # Invert: shorter best duration is better
    bd_max = best_durs.max()
    if bd_max > 0:
        scores += 0.3 * (1.0 - best_durs / bd_max)
    
    # Component 4: Soft-min over all machine options (captures spread)
    # Use log-sum-exp based soft-min for smoothness
    soft_min_scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durs = np.array(list(mo.values()), dtype=np.float64)
            # Soft-min: -logsumexp(-durs) approximates min but is smooth
            # For small arrays, direct computation is fine
            neg_durs = -durs
            max_neg = neg_durs.max()
            if max_neg < 1e10:  # avoid overflow
                soft_min_val = -max_neg - np.log(np.sum(np.exp(neg_durs - max_neg)))
            else:
                soft_min_val = durs.min()
            soft_min_scores[i] = soft_min_val
        else:
            soft_min_scores[i] = 0.0
    
    # Invert soft-min: smaller soft-min (closer to true min) is better
    sm_max = soft_min_scores.max()
    if sm_max > 0:
        scores += 0.2 * (1.0 - soft_min_scores / sm_max)
    
    # Component 5: Number of machine options (more flexibility is slightly better)
    n_opts = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    no_max = n_opts.max()
    if no_max > 0:
        scores += 0.1 * (n_opts / no_max)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
