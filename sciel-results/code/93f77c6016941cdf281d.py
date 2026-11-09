"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = None
_DEADLINE = 140.0
_SOFT_MIN_T = 0.1  # Temperature for soft-min

def _soft_min(values, t=_SOFT_MIN_T):
    """Compute soft-min of a list of values using log-sum-exp trick."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    min_val = min(values)
    # Soft-min = -log(sum(exp(-v/t)) * t)
    # = min_val - t * log(sum(exp(-(v - min_val)/t)))
    scaled = [-(v - min_val) / t for v in values]
    max_scaled = max(scaled)
    log_sum = max_scaled + math.log(sum(math.exp(s - max_scaled) for s in scaled))
    return min_val - t * log_sum

def dispatch(ready_operations, time):
    """
    Job-Priority Dispatch Rule with Weighted Machine Load Estimation.
    
    Strategy:
    1. Prioritize operations based on the total remaining work of their respective jobs (LRPT).
    2. Use a soft-min based cost that incorporates weighted distribution across ALL machine options
       to better capture machine congestion patterns.
    3. Select the machine that minimizes the estimated completion time considering both
       the operation duration and the relative load on that machine.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        if not ready_operations:
            return 0
        # Fallback: First operation
        return 0
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    if n_ops == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            # Pick shortest duration machine
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Extract features and compute weighted machine load estimate
    features = []
    for i, op in enumerate(ready_operations):
        job = op.get("job", i)
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 1)
        mo = op.get("machine_options", {})
        
        if mo:
            durations = list(mo.values())
            machines = list(mo.keys())
            min_d = min(durations)
            
            # Weighted load estimate: use soft-min across all machine durations
            # This gives a smoother estimate than hard min, reflecting that
            # even if one machine is fast, others may be congested
            soft_min_d = _soft_min(durations, _SOFT_MIN_T)
            
            # Also compute mean as a secondary congestion signal
            mean_d = sum(durations) / len(durations)
            
            # Composite cost: blend of soft-min and mean to capture both
            # the best-case and average-case machine availability
            # Weight: 0.7 * soft_min + 0.3 * mean
            composite_d = 0.7 * soft_min_d + 0.3 * mean_d
            
            # Select machine: minimize duration, but penalize if the machine
            # is significantly above the soft-min (indicating congestion)
            # Use a simple approach: pick the machine with minimum duration,
            # but if there's a tie or near-tie, prefer the one closer to soft-min
            best_machine = min(mo, key=mo.get)
            
            # Check if there's a better machine considering congestion
            # If the min-duration machine is much slower than soft-min suggests,
            # another machine might be better
            min_machine_dur = mo[best_machine]
            if min_machine_dur > soft_min_d * 1.5:
                # Look for a machine with duration closer to soft_min
                alt_machine = min(mo, key=lambda m: abs(mo[m] - soft_min_d))
                if mo[alt_machine] < min_machine_dur * 1.2:
                    best_machine = alt_machine
        else:
            min_d = 1.0
            composite_d = 1.0
            best_machine = None
            
        features.append({
            'index': i,
            'job': job,
            'rw': rw,
            'ro': ro,
            'min_d': min_d,
            'composite_d': composite_d,
            'best_machine': best_machine
        })
    
    # Sort by:
    # 1. Job priority: higher remaining_work first (LRPT)
    # 2. Lower composite machine load cost
    # 3. Job ID for determinism
    sorted_features = sorted(features, key=lambda x: (-x['rw'], x['composite_d'], x['job']))
    
    best_feat = sorted_features[0]
    
    if best_feat['best_machine'] is not None:
        return (best_feat['index'], best_feat['best_machine'])
    else:
        return best_feat['index']

# EVOLVE-BLOCK-END
