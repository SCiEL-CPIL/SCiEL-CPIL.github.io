import numpy as np
import time as _time_module

# EVOLVE-BLOCK-START
_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0


def dispatch(ready_operations, time):
    """
    Dispatch rule: Critical-path slack priority with WSPT refinement.
    
    Structural change: Model each ready operation's urgency via "slack" -
    the difference between remaining work and elapsed time, normalized by
    the minimum available processing duration. This captures how tight the
    schedule is for each job's critical path.
    
    Primary: slack_ratio = max(remaining_work - time, 0) / min_duration
             (lower slack_ratio = more urgent = higher priority)
    Secondary: remaining_ops / min_duration (WSPT, maximize)
    
    Combined: minimize slack_ratio, break ties by maximizing WSPT.
    """
    if not ready_operations:
        return 0

    if _time_module.perf_counter() > _DEADLINE:
        return 0

    n_ops = len(ready_operations)
    
    # Compute min_duration for each operation
    min_durations = np.zeros(n_ops, dtype=np.float64)
    remaining_works = np.zeros(n_ops, dtype=np.float64)
    remaining_ops_arr = np.zeros(n_ops, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            min_durations[i] = 1.0
        else:
            durations = list(machine_options.values())
            min_durations[i] = min(durations) if durations else 1.0
            if min_durations[i] <= 0:
                min_durations[i] = 1e-9
        
        rw = op.get("remaining_work", 0)
        remaining_works[i] = rw if rw is not None else 0.0
        
        ro = op.get("remaining_ops", 1)
        remaining_ops_arr[i] = ro if ro > 0 else 1
    
    # Critical path slack: how much "buffer" does each job have?
    # slack = remaining_work - time (how much work is left relative to elapsed time)
    # If slack is small or negative, the job is on a tight critical path
    slack = remaining_works - time
    
    # Clamp to avoid negative slack dominating (negative slack = already behind)
    # We want to prioritize operations with LOW slack (most urgent)
    # But we also need to consider the processing time to get that work done
    slack_ratio = np.maximum(slack, 0.0) / min_durations
    
    # WSPT score: remaining_ops / min_duration (maximize)
    wspt_score = remaining_ops_arr / min_durations
    
    # Normalize both to [0, 1] for combination
    # slack_ratio: lower is better (more urgent)
    sr_min, sr_max = slack_ratio.min(), slack_ratio.max()
    if sr_max - sr_min < 1e-12:
        slack_norm = np.ones(n_ops) * 0.5
    else:
        # Invert: lower slack_ratio -> higher norm
        slack_norm = 1.0 - (slack_ratio - sr_min) / (sr_max - sr_min)
    
    # wspt_score: higher is better
    ws_min, ws_max = wspt_score.min(), wspt_score.max()
    if ws_max - ws_min < 1e-12:
        wspt_norm = np.ones(n_ops) * 0.5
    else:
        wspt_norm = (wspt_score - ws_min) / (ws_max - ws_min)
    
    # Weighted combination: slack urgency dominates, WSPT refines
    # The slack term captures "how tight is this job's schedule"
    # The WSPT term captures "how much downstream work depends on this"
    total_score = 0.7 * slack_norm + 0.3 * wspt_norm
    
    # Additional tie-break: prefer jobs with fewer remaining operations
    # (closer to completion) when scores are very close
    best_idx = int(np.argmax(total_score))
    
    # Check for near-ties and break by remaining_ops (lower = closer to done)
    max_score = total_score[best_idx]
    tie_mask = np.abs(total_score - max_score) < 1e-9
    if np.sum(tie_mask) > 1:
        tie_indices = np.where(tie_mask)[0]
        # Among ties, pick the one with lowest remaining_ops (closest to completion)
        tie_remaining_ops = remaining_ops_arr[tie_indices]
        best_in_tie = tie_indices[np.argmin(tie_remaining_ops)]
        best_idx = int(best_in_tie)
    
    return best_idx
# EVOLVE-BLOCK-END
