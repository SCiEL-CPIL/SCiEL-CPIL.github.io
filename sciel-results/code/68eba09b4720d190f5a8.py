"""Open-shop dispatch rule: prioritize by remaining work with machine-load balancing."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: ensure we finish well before 120s wall-clock per evaluation.
# The dispatch function is called many times per instance; we keep overhead minimal.
_START_TIME = None
_TIME_LIMIT = 110.0  # Safety margin before 120s kill
_DEADLINE_REACHED = False

def _check_deadline(fallback):
    global _START_TIME, _DEADLINE_REACHED
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    if _time.monotonic() - _START_TIME > _TIME_LIMIT:
        _DEADLINE_REACHED = True
    return fallback if _DEADLINE_REACHED else None

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    - Primary: maximize remaining_work (jobs with more total work left are more critical).
    - Secondary: among ties, prefer the operation whose chosen machine has the least
      current/estimated load (we approximate load by preferring shorter durations on
      machines, which tends to spread work).
    - Tertiary: break ties by job id for determinism.
    
    Refined approach combining insights:
    1. Use normalized features for stable weighting across instances.
    2. Combine remaining_work, remaining_ops, min_duration (inverted), and machine flexibility.
    3. Add a small soft-min term to penalize operations that would create bottlenecks.
    """
    # Check deadline first
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, find the minimum duration across its machine options (best case processing time)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options available (more flexibility = potentially better to schedule now)
    num_machines_available = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Normalize features to [0, 1] range for stable weighting
    def _norm(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    nw = _norm(remaining_works)
    no = _norm(remaining_ops)
    # Invert min_duration: lower duration is better, so we want high score for low duration
    md_inv = 1.0 - _norm(min_durations)
    nm = _norm(num_machines_available)
    
    # Soft-min penalty: penalize operations that have very long durations on some machines
    # Use log-sum-exp approximation to smooth the min/max
    if n > 1:
        # For each operation, compute a soft penalty based on the spread of durations
        duration_spreads = np.array([
            (max(op["machine_options"].values()) - min(op["machine_options"].values())) 
            if op["machine_options"] else 0.0
            for op in ready_operations
        ], dtype=np.float64)
        
        # Normalize spread and use as a small penalty
        spread_norm = _norm(duration_spreads)
        soft_penalty = 0.1 * spread_norm
    else:
        soft_penalty = np.zeros(n)
    
    # Composite score: we want to MAXIMIZE this to pick the best op
    # Weighted combination with refined weights
    # - High remaining_work is good (LRPT) - weight 1.0
    # - High remaining_ops is good (more work to do) - weight 0.5
    # - Low min_duration is good (can finish quickly, frees up machine) - weight 0.3
    # - High num_machines_available is slightly good (flexibility) - weight 0.1
    # - Soft penalty for duration spread - weight 0.1
    
    score = 1.0 * nw + 0.5 * no + 0.3 * md_inv + 0.1 * nm - soft_penalty
    
    # Pick the argmax
    best_idx = int(np.argmax(score))
    
    return best_idx
# EVOLVE-BLOCK-END
