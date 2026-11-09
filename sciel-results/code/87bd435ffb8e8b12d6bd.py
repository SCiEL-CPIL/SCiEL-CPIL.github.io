import time as _time
import math

_START_TIME = None
_DEADLINE = 150.0

def _soft_min(values, temperature=0.5):
    """Compute soft minimum using log-sum-exp trick."""
    if not values:
        return 1.0
    if len(values) == 1:
        return values[0]
    
    # For soft min: -soft_max(-values)
    # soft_max(x) = log(sum(exp(x_i / T))) * T
    # soft_min(x) = -log(sum(exp(-x_i / T))) * T
    
    neg_vals = [-v for v in values]
    max_neg = max(neg_vals)
    if max_neg > 500:  # overflow protection
        return min(values)
    
    exp_sum = sum(math.exp((nv - max_neg) / temperature) for nv in neg_vals)
    soft_min_val = -(max_neg + math.log(exp_sum)) * temperature
    
    # Ensure it's not negative
    return max(0.1, soft_min_val)

def _weighted_machine_selection(mo, temperature=0.3):
    """Select machine using soft-min weighted scoring with tie-breaking."""
    if not mo:
        return None
    if len(mo) == 1:
        return list(mo.keys())[0]
    
    # Use soft-min as a smooth surrogate for the minimum duration
    # Lower soft-min means more "efficient" machine options available
    durations = list(mo.values())
    sm = _soft_min(durations, temperature=temperature)
    
    # For machine selection, prefer the machine with shortest duration
    # but use soft-min to break ties smoothly
    best_machine = min(mo, key=mo.get)
    best_dur = mo[best_machine]
    
    # If there are multiple machines with similar durations, 
    # the soft-min will be close to the minimum, indicating flexibility
    # We can use this to slightly prefer machines that are part of a "cluster"
    # of short-duration options
    
    # For now, just return the best machine (shortest duration)
    # The soft-min is used in the overall scoring
    return best_machine

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    if _time.time() - _START_TIME > _DEADLINE - 5.0:
        if not ready_operations:
            return 0
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            # Pick shortest machine
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Pre-calculate features
    max_rw = 0.0
    min_min_d = float('inf')
    max_min_d = 0.0
    min_n_machines = float('inf')
    sum_rw = 0.0
    sum_rw_sq = 0.0
    sum_soft_min = 0.0
    sum_remaining_ops = 0.0
    max_remaining_ops = 0.0
    sum_flex_ratio = 0.0
    
    features = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        rem_ops = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            n_machines = len(mo)
            best_machine = min(mo, key=mo.get)
            # Soft minimum for smoother machine selection
            sm = _soft_min(durations, temperature=0.3)
            # Also compute a "flexibility score" - how close the other machines are to the best
            # This helps avoid over-constraining to a single machine
            if n_machines > 1:
                sorted_durs = sorted(durations)
                # Ratio of second-best to best (closer to 1 means more flexible)
                flex_ratio = sorted_durs[1] / max(sorted_durs[0], 0.1)
            else:
                flex_ratio = 1.0
        else:
            min_d = 1.0
            n_machines = 1
            best_machine = None
            sm = 1.0
            flex_ratio = 1.0

        features.append({
            'rw': rw,
            'min_d': min_d,
            'soft_min_d': sm,
            'n_machines': n_machines,
            'best_machine': best_machine,
            'remaining_ops': rem_ops,
            'flex_ratio': flex_ratio
        })
        
        if rw > max_rw:
            max_rw = rw
        if min_d < min_min_d:
            min_min_d = min_d
        if min_d > max_min_d:
            max_min_d = min_d
        if n_machines < min_n_machines:
            min_n_machines = n_machines
        if rem_ops > max_remaining_ops:
            max_remaining_ops = rem_ops
            
        sum_rw += rw
        sum_rw_sq += rw * rw
        sum_soft_min += sm
        sum_remaining_ops += rem_ops
        sum_flex_ratio += flex_ratio

    # Handle edge cases for normalization
    if max_rw == 0: 
        max_rw = 1.0
        mean_rw = 1.0
        std_rw = 1.0
    else:
        mean_rw = sum_rw / n
        variance_rw = (sum_rw_sq / n) - (mean_rw * mean_rw)
        if variance_rw < 0:
            variance_rw = 0.0
        std_rw = math.sqrt(variance_rw)
        
    if min_min_d == 0: min_min_d = 1.0
    if max_min_d == 0: max_min_d = 1.0
    if min_n_machines == 0: min_n_machines = 1.0
    if max_remaining_ops == 0: max_remaining_ops = 1.0
    
    mean_soft_min = sum_soft_min / n if n > 0 else 1.0
    if mean_soft_min == 0:
        mean_soft_min = 1.0
    mean_flex_ratio = sum_flex_ratio / n if n > 0 else 1.0

    # Dynamic Weights based on variance of Remaining Work
    if mean_rw > 1e-6:
        var_factor = std_rw / mean_rw
    else:
        var_factor = 0.0
        
    # Clamp var_factor to [0, 1] for stable weighting
    if var_factor > 1.0:
        var_factor = 1.0
    elif var_factor < 0.0:
        var_factor = 0.0

    # Weights:
    # High variance -> w_rw high, w_eff low
    # Low variance -> w_rw low, w_eff high
    # Base weights with emphasis on remaining work and efficiency
    w_rw = 0.5 + 0.3 * var_factor   # Range: [0.5, 0.8]
    w_eff = 0.4 - 0.2 * var_factor  # Range: [0.2, 0.4]
    w_flex = 0.1                    # Fixed small weight for flexibility/constraint
    w_ops = 0.1                     # Weight for remaining operations count
    
    # Normalize weights so they sum to 1.0
    total_w = w_rw + w_eff + w_flex + w_ops
    w_rw /= total_w
    w_eff /= total_w
    w_flex /= total_w
    w_ops /= total_w

    best_score = -1.0
    best_idx = 0
    best_machine = None

    for i in range(n):
        f = features[i]
        
        # Component 1: Remaining Work (Higher is better)
        # Non-linear scaling to emphasize larger differences
        if f['rw'] == 0:
            score_rw = 0.0
        else:
            score_rw = (f['rw'] / max_rw) ** 1.2
        
        # Component 2: Efficiency using soft minimum (Lower is better)
        # Inverse normalize with soft min for smoother behavior
        score_eff = (min_min_d / f['soft_min_d']) ** 1.3
        
        # Component 3: Flexibility (Lower n_machines is better)
        # Inverse normalize
        score_flex = min_n_machines / f['n_machines']
        
        # Component 4: Remaining Operations (Higher is better - more work to do)
        if max_remaining_ops == 0:
            score_ops = 0.0
        else:
            score_ops = f['remaining_ops'] / max_remaining_ops
        
        # Component 5: Machine flexibility ratio (closer to 1 means more balanced options)
        # This is a smooth surrogate that rewards having multiple competitive machines
        score_flex_ratio = 1.0 / (1.0 + max(0.0, f['flex_ratio'] - 1.0))
        
        total_score = w_rw * score_rw + w_eff * score_eff + w_flex * score_flex + w_ops * score_ops + 0.05 * score_flex_ratio
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = f['best_machine']
        elif abs(total_score - best_score) < 1e-9:
            # Tie-breaker: Prefer shorter duration
            if f['min_d'] < features[best_idx]['min_d']:
                best_idx = i
                best_machine = f['best_machine']
            elif abs(f['min_d'] - features[best_idx]['min_d']) < 1e-9:
                # Secondary tie-breaker: Prefer higher flexibility ratio
                if f['flex_ratio'] > features[best_idx]['flex_ratio']:
                    best_idx = i
                    best_machine = f['best_machine']

    # Return (index, machine)
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END
