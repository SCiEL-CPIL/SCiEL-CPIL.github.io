"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard
_DEADLINE = _time.monotonic() + 110.0  # safety margin before 120s kill


def _min_duration(op):
    """Return the minimum processing time among machine options."""
    return min(op["machine_options"].values())


def _critical_path_length(op):
    """
    Estimate the remaining critical path length for a ready operation.
    This is the sum of minimum processing times for all subsequent operations
    in the job (precedence-aware urgency).
    """
    # remaining_ops is the number of operations still to be processed in this job
    # including the current ready one? Or just subsequent?
    # The field is "remaining_ops" - let's assume it counts ops yet to be scheduled
    # including the current one. We need the sum of min durations for those ops.
    # But we only have aggregate "remaining_work" and "remaining_ops".
    # We don't have per-op durations for future ops in the ready_operation dict.
    # However, remaining_work / remaining_ops gives an average min-ish duration.
    # A better proxy: remaining_work is the total work, but we want critical path.
    # Since we don't have individual future op durations, we approximate:
    # critical_path ≈ remaining_work (if we assume serial execution on min machines)
    # But the directive says "sum of minimum processing times for all subsequent operations".
    # We don't have that data directly. Let's use what we have.
    
    # Actually, looking at the interface: each ready operation has "remaining_ops" and "remaining_work".
    # remaining_work is likely the sum of min durations for all remaining ops in the job.
    # So remaining_work IS the critical path length if each op takes its min duration.
    return op["remaining_work"]


def _machine_load_estimate(op, time):
    """
    Estimate how contested the machine options are.
    We don't have direct machine load info, so we use a heuristic:
    prefer operations where the min-duration machine is less likely to be busy.
    Since we can't query machine state, we use a simple heuristic based on
    the number of machine options (more options = less contestation risk).
    """
    n_options = len(op["machine_options"])
    # More options means less likely to be blocked, so lower contestation
    return 1.0 / (n_options + 1)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next using a composite priority:
    1. Critical path length (remaining work) - higher urgency
    2. Duration minimization - prefer shorter operations to reduce tail
    3. Machine contestation avoidance - prefer operations with more options
    
    Returns index of the ready operation to start.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Check deadline
    if _time.monotonic() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    # We want to MAXIMIZE a priority score (higher = more urgent)
    
    # 1. Critical path / remaining work (higher = more urgent)
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    # 2. Minimum duration of the current operation (lower = better, so we negate or invert)
    min_durations = np.array([_min_duration(op) for op in ready_operations], dtype=np.float64)
    
    # 3. Number of machine options (higher = less contestation)
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    
    # Normalize each component to [0, 1] range for fair combination
    def _normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_work = _normalize(remaining_works)
    norm_dur = _normalize(min_durations)  # higher norm_dur means longer, so we want to penalize
    norm_opts = _normalize(n_options)
    
    # Composite priority:
    # - High remaining work = high urgency (positive weight)
    # - Low duration = good (we want to prefer short ops, so negative weight on duration)
    # - More options = less contestation (positive weight)
    
    # Weighted combination:
    # Priority = w1 * norm_work - w2 * norm_dur + w3 * norm_opts
    # We want to start operations with high urgency, short processing, and flexible machine choice.
    
    w1 = 0.5  # urgency weight
    w2 = 0.3  # duration penalty weight
    w3 = 0.2  # flexibility weight
    
    priorities = w1 * norm_work - w2 * norm_dur + w3 * norm_opts
    
    # Select the operation with the highest priority
    best_idx = int(np.argmax(priorities))
    
    return best_idx
# EVOLVE-BLOCK-END
