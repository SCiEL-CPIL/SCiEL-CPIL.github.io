"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.time() + 140  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule: pick the ready operation to start next.
    Uses a multi-criteria priority score combining:
      1. Remaining work (LRPT-like)
      2. Critical ratio urgency (remaining_work / remaining_ops)
      3. Load-aware penalty (prefer less-loaded machines)
      4. Urgency bonus for operations with few remaining ops
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _time.time() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # Avoid division by zero
    remaining_ops_safe = np.maximum(remaining_ops, 1.0)

    # Critical ratio: work per remaining op (higher = more urgent per step)
    crit_ratio = remaining_works / remaining_ops_safe

    # Machine load: for each ready op, compute the minimum duration across machine options
    # and also the average duration to gauge how "expensive" this op is
    min_durs = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=np.float64)
    avg_durs = np.array([np.mean(list(op["machine_options"].values())) for op in ready_operations], dtype=np.float64)

    # Normalize each component to [0, 1] range for fair combination
    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    norm_work = _norm(remaining_works)
    norm_crit = _norm(crit_ratio)
    norm_min_dur = _norm(min_durs)
    norm_avg_dur = _norm(avg_durs)

    # Urgency bonus: operations with fewer remaining ops are more urgent
    # (they're closer to completion, so delaying them hurts more)
    urgency_bonus = _norm(1.0 / remaining_ops_safe)

    # Load-aware penalty: prefer ops that can run on less-loaded machines
    # (lower min_duration suggests the op is lighter, less bottleneck)
    # We use a moderate penalty so it doesn't dominate
    load_penalty = norm_min_dur * 0.3 + norm_avg_dur * 0.1

    # Combined score
    # Weighting: remaining work is the primary driver, critical ratio adds urgency,
    # urgency bonus helps break ties, load penalty is a soft tiebreaker
    score = (
        1.0 * norm_work +
        0.5 * norm_crit +
        0.4 * urgency_bonus -
        0.2 * load_penalty
    )

    idx = int(np.argmax(score))
    return idx
# EVOLVE-BLOCK-END
