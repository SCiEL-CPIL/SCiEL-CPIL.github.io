"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard
_START_TIME = _time.time()
_HARD_LIMIT = 110.0
_SOFT_MARGIN = 5.0

def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def _normalize(arr):
    """Min-max normalize to [0,1]. If all same, return zeros."""
    a = np.asarray(arr, dtype=np.float64)
    mn = a.min()
    mx = a.max()
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic combining multiple scheduling rules to minimize makespan.
    
    Heuristics considered:
    1. Most remaining work (MRW) - critical path
    2. Minimum remaining processing time on assigned machine
    3. Most operations remaining (MOR)
    4. Critical path-like estimate
    5. Machine constraint tightness
    
    We use a weighted combination with proper normalization that tends to favor
    operations with high remaining work and few machine options (tighter constraints first).
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if too close, fall back to fast simple rule
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features into arrays for vectorized computation
    rem_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    rem_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    min_durs = np.zeros(n, dtype=np.float64)
    avg_durs = np.zeros(n, dtype=np.float64)
    num_machines = np.ones(n, dtype=np.int32)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            min_durs[i] = min(durations)
            avg_durs[i] = sum(durations) / len(durations)
            num_machines[i] = len(durations)
    
    # Heuristic 1: Most remaining work (normalized)
    h1 = _normalize(rem_works)
    
    # Heuristic 2: Most remaining operations (urgency, normalized)
    h2 = _normalize(rem_ops)
    
    # Heuristic 3: Shorter minimum processing time (finish sooner to free machine)
    # Inverse with scaling, then normalize
    h3_raw = 1.0 / (min_durs + 1.0)
    h3 = _normalize(h3_raw)
    
    # Heuristic 4: Fewer machine options means more constrained, prioritize
    h4_raw = 1.0 / (num_machines.astype(np.float64) + 1.0)
    h4 = _normalize(h4_raw)
    
    # Heuristic 5: Ratio of remaining work to average duration (urgency of long jobs)
    h5_raw = rem_works / (avg_durs + 1.0)
    h5 = _normalize(h5_raw)
    
    # Heuristic 6: Critical path estimate - remaining work weighted by remaining ops
    h6_raw = rem_works * (rem_ops + 1.0)
    h6 = _normalize(h6_raw)
    
    # Heuristic 7: Soft-max approximation of critical path using log-sum-exp
    # This provides a smooth surrogate for the max processing time across machines
    # Higher values indicate more "spread" in processing times, which can indicate
    # scheduling flexibility or constraint
    with np.errstate(divide='ignore', invalid='ignore'):
        log_durs = np.log(min_durs + 1e-9)
        # Use a temperature parameter to control smoothness
        temp = 0.5
        log_sum_exp = np.log(np.exp(log_durs / temp)).sum(axis=-1) if len(log_durs.shape) > 1 else np.log(np.exp(log_durs / temp))
        # Actually, let's use a simpler approach: weighted average with exponential weights
        # This gives more weight to longer durations (critical path)
        exp_weights = np.exp(min_durs / temp)
        h7_raw = (exp_weights * min_durs).sum() / (exp_weights.sum() + 1e-9) if n > 0 else 0.0
        # For individual operations, just use the max duration as a proxy
        h7_raw = np.zeros(n)
        for i, op in enumerate(ready_operations):
            mo = op["machine_options"]
            if mo:
                h7_raw[i] = max(mo.values())
        h7 = _normalize(h7_raw)
    
    # Combine heuristics with weights tuned for open-shop makespan minimization
    # Primary: most remaining work (critical path)
    # Secondary: urgency (fewer ops left), speed (short min dur), constraint
    # Tertiary: critical path estimate, max duration
    weights = np.array([1.5, 0.8, 0.4, 0.3, 0.7, 1.0, 0.3], dtype=np.float64)
    
    scores = (
        weights[0] * h1 +
        weights[1] * h2 +
        weights[2] * h3 +
        weights[3] * h4 +
        weights[4] * h5 +
        weights[5] * h6 +
        weights[6] * h7
    )
    
    # Add tiny random tiebreaker to avoid deterministic bias in close cases
    scores += np.random.uniform(0, 0.001, size=n)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
