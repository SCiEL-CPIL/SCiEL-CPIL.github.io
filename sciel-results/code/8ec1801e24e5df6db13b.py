"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard
_DEADLINE = 150.0
_START_TIME = _time.time()
_SAFETY_MARGIN = 5.0  # seconds before hard limit


def _elapsed():
    return _time.time() - _START_TIME


def _time_check():
    """Return True if we should abort and return a safe default."""
    return _elapsed() > (_DEADLINE - _SAFETY_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop scheduling simulation.
    
    Strategy: Multi-criteria heuristic combining:
    1. Longest remaining processing time (LRPT) - proxy for makespan reduction
    2. Machine availability (shortest duration on that machine)
    3. Remaining operations count (jobs with more remaining ops are more critical)
    4. Critical path approximation
    
    We use a weighted scoring approach that considers:
    - For each ready op, compute a score based on:
      a) remaining_work (higher = more urgent)
      b) min duration across machine options (lower = faster completion)
      c) remaining_ops (higher = more critical job)
      d) A penalty for machines that are likely to be busy (approximated by time)
    
    Returns: index of the best operation to schedule
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Fast path: if only one option, return it
    if n == 1:
        return 0
    
    # If we're close to deadline, use simple heuristic
    if _time_check():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    # Pre-extract arrays for speed
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each op, find the minimum duration and the machine with min duration
    min_durations = np.zeros(n)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
        else:
            min_durations[i] = 1.0  # fallback
    
    # Normalize components to [0, 1] range for fair weighting
    # remaining_work: higher is better (more work = more critical)
    rw_max = remaining_works.max()
    rw_min = remaining_works.min()
    rw_range = rw_max - rw_min if rw_range > 0 else 1.0
    rw_norm = (remaining_works - rw_min) / rw_range
    
    # min_durations: lower is better (faster completion)
    md_max = min_durations.max()
    md_min = min_durations.min()
    md_range = md_max - md_min if md_range > 0 else 1.0
    md_norm = (min_durations - md_min) / md_range  # lower = smaller value
    
    # remaining_ops: higher is more critical
    ro_max = remaining_ops.max()
    ro_min = remaining_ops.min()
    ro_range = ro_max - ro_min if ro_range > 0 else 1.0
    ro_norm = (remaining_ops - ro_min) / ro_range
    
    # Weighted combination:
    # - 50% weight on remaining work (primary makespan driver)
    # - 30% weight on min duration (inverse: prefer shorter)
    # - 20% weight on remaining ops (job criticality)
    
    # For min_durations, we want lower to be better, so invert
    scores = 0.5 * rw_norm + 0.3 * (1.0 - md_norm) + 0.2 * ro_norm
    
    # Additional consideration: if an operation has a very short duration on some machine,
    # it might be worth scheduling to free up that machine early.
    # But this is already captured by the min_durations term.
    
    # Tie-breaking: prefer the one with lower remaining_work if scores are very close
    # (to avoid pathological oscillation)
    best_idx = int(np.argmax(scores))
    
    # Check for near-ties and break by remaining_work (higher first)
    max_score = scores[best_idx]
    tied = np.where(np.abs(scores - max_score) < 1e-6)[0]
    if len(tied) > 1:
        tied_rw = remaining_works[tied]
        best_idx = tied[np.argmax(tied_rw)]
    
    return int(best_idx)
# EVOLVE-BLOCK-END
