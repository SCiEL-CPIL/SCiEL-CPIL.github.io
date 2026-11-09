"""Combined Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit

# Pre-compute weights for different strategies
_W_MWL = 0.35       # Minimum Work Load (remaining work)
_W_SPT = 0.30       # Shortest Processing Time (inverse min duration)
_W_CPL = 0.20       # Critical Path Length proxy (remaining_work / remaining_ops)
_W_FLEX = 0.15      # Flexibility (number of machine options / spread)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Combine multiple heuristics into a composite score:
    1. Critical Path Length (CPL): Jobs with longer remaining paths are urgent.
    2. Minimum Remaining Work: Favors operations that finish quickly (shortest processing time).
    3. Machine Load Balance: Prefer less busy machines if we can infer from options.
    4. Number of remaining ops: More remaining ops = higher priority (tail of the job).
    
    The composite score is:
      score = w1 * normalized_remaining_work + 
              w2 * (1 / (min_duration + epsilon)) + 
              w3 * (1 / (remaining_ops + 1))
    
    We pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features using numpy for efficiency
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, find the minimum duration, max duration, and count among machine options
    min_durations = np.zeros(n, dtype=np.float64)
    max_durations = np.zeros(n, dtype=np.float64)
    option_counts = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            vals = list(mo.values())
            min_durations[i] = min(vals)
            max_durations[i] = max(vals)
            option_counts[i] = len(vals)
        else:
            min_durations[i] = 1.0  # fallback
            max_durations[i] = 1.0
            option_counts[i] = 1.0

    # Normalize remaining_work to [0, 1]
    max_rw = remaining_works.max()
    min_rw = remaining_works.min()
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = np.full(n, 0.5)
    else:
        norm_rw = (remaining_works - min_rw) / rw_range

    # Inverse min_duration: shorter processing time gets higher score
    inv_dur = 1.0 / (min_durations + 1e-6)
    max_inv = inv_dur.max()
    if max_inv < 1e-9:
        norm_inv_dur = np.full(n, 0.5)
    else:
        norm_inv_dur = inv_dur / max_inv

    # Critical Path Length proxy: remaining_work / remaining_ops
    # This identifies jobs with high average processing times per remaining op
    with np.errstate(divide='ignore', invalid='ignore'):
        cpl_proxy = remaining_works / (remaining_ops + 1e-6)
    max_cpl = cpl_proxy.max()
    min_cpl = cpl_proxy.min()
    cpl_range = max_cpl - min_cpl
    if cpl_range < 1e-9:
        norm_cpl = np.full(n, 0.5)
    else:
        norm_cpl = (cpl_proxy - min_cpl) / cpl_range

    # Flexibility: more machine options and lower spread (max-min) is better
    # Normalize option counts
    max_opts = option_counts.max()
    min_opts = option_counts.min()
    opts_range = max_opts - min_opts
    if opts_range < 1e-9:
        norm_opts = np.full(n, 0.5)
    else:
        norm_opts = (option_counts - min_opts) / opts_range
    
    # Spread: lower (max-min) duration spread is better
    spread = max_durations - min_durations
    max_spread = spread.max()
    min_spread = spread.min()
    spread_range = max_spread - min_spread
    if spread_range < 1e-9:
        norm_spread = np.full(n, 0.5)
    else:
        norm_spread = (spread - min_spread) / spread_range
    # Invert spread: lower spread is better
    norm_spread_inv = 1.0 - norm_spread

    # Combine flexibility: balance of option count and low spread
    norm_flex = 0.5 * norm_opts + 0.5 * norm_spread_inv

    # Composite score with tuned weights
    scores = (_W_MWL * norm_rw + 
              _W_SPT * norm_inv_dur + 
              _W_CPL * norm_cpl + 
              _W_FLEX * norm_flex)

    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
