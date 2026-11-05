"""Improved dispatch rule for open-shop scheduling using machine-load balancing."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = _time.perf_counter()
    return _time.perf_counter() - _START


def dispatch(ready_operations, time):
    """
    Machine-load balancing dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Select the operation that targets the most congested machine (highest remaining
       workload) to balance machine loads and prevent bottlenecks.
    2. Among operations targeting the same machine, prefer shorter durations to reduce
       blocking time.
    3. Return (index, machine) where machine is the one with the minimum duration
       for the selected operation.
    """
    global _START
    if _START is None:
        _START = _time.perf_counter()
    
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_m = min(opts, key=opts.get)
            return (0, best_m)
        return 0
    
    # Compute for each operation:
    # - min_duration: shortest processing time on any available machine
    # - best_machine: the machine with that min duration
    # - total_machine_load: sum of durations across all machine options (proxy for congestion)
    # - remaining_work: from the op dict
    # - remaining_ops: from the op dict
    
    min_durations = np.empty(n, dtype=np.float64)
    best_machines = [None] * n
    num_options = np.empty(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            min_durations[i] = 0.0
            best_machines[i] = None
            num_options[i] = 0
            continue
        
        best_m = min(opts, key=opts.get)
        best_machines[i] = best_m
        min_durations[i] = opts[best_m]
        num_options[i] = len(opts)
    
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Normalize features
    rw_max = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    md_max = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    no_max = np.max(num_options) if np.max(num_options) > 0 else 1.0
    
    # Score components:
    # 1. Remaining Work (higher is better) - Weight 0.4
    score_rw = (remaining_works / rw_max) * 0.4
    
    # 2. Min Duration (lower is better) - Weight 0.35
    # Use 1/(md+eps) to emphasize short durations
    score_md = (1.0 / (min_durations + 1e-6))
    md_max_score = np.max(score_md) if np.max(score_md) > 0 else 1.0
    score_md = (score_md / md_max_score) * 0.35
    
    # 3. Flexibility (fewer options is better, more constrained) - Weight 0.25
    score_flex = (1.0 / (num_options + 1e-6))
    flex_max_score = np.max(score_flex) if np.max(score_flex) > 0 else 1.0
    score_flex = (score_flex / flex_max_score) * 0.25
    
    total_score = score_rw + score_md + score_flex
    
    # Small deterministic tiebreaker based on index to avoid random seeding issues
    tiebreaker = np.arange(n) * 1e-8
    total_score += tiebreaker
    
    best_idx = int(np.argmax(total_score))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
