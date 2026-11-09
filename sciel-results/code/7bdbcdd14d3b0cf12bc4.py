import time as _time
import math

_START_TIME = None
_DEADLINE = 150.0

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
    min_remaining_ops = float('inf')
    
    features = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        remaining_ops = op.get("remaining_ops", 0)
        
        if mo:
            durations = list(mo.values())
            min_d = min(durations)
            n_machines = len(mo)
            best_machine = min(mo, key=mo.get)
        else:
            min_d = 1.0
            n_machines = 1
            best_machine = None

        features.append({
            'rw': rw,
            'min_d': min_d,
            'n_machines': n_machines,
            'best_machine': best_machine,
            'remaining_ops': remaining_ops
        })
        
        if rw > max_rw:
            max_rw = rw
        if min_d < min_min_d:
            min_min_d = min_d
        if min_d > max_min_d:
            max_min_d = min_d
        if n_machines < min_n_machines:
            min_n_machines = n_machines
        if remaining_ops < min_remaining_ops:
            min_remaining_ops = remaining_ops
            
        sum_rw += rw
        sum_rw_sq += rw * rw

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
    if min_remaining_ops == 0: min_remaining_ops = 1.0

    # Dynamic Weights based on variance of Remaining Work
    # If variance is high, MWR (remaining work) is a strong discriminator -> weight it more.
    # If variance is low, all jobs are similar size -> use Efficiency (min duration) to break ties.
    
    # Normalize variance factor to [0, 1] roughly
    # A variance factor of 1 means std is equal to mean (high spread).
    # A variance factor of 0 means no spread.
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
    # Base weights
    w_rw = 0.4 + 0.4 * var_factor   # Range: [0.4, 0.8]
    w_eff = 0.6 - 0.4 * var_factor  # Range: [0.2, 0.6]
    w_flex = 0.1                    # Fixed small weight for flexibility/constraint
    
    # Normalize weights so they sum to 1.0 (optional but good for consistency)
    total_w = w_rw + w_eff + w_flex
    w_rw /= total_w
    w_eff /= total_w
    w_flex /= total_w

    best_score = -1.0
    best_idx = 0
    best_machine = None

    for i in range(n):
        f = features[i]
        
        # Component 1: Remaining Work (Higher is better)
        # Non-linear scaling (power 1.5) to emphasize larger differences
        if f['rw'] == 0:
            score_rw = 0.0
        else:
            score_rw = (f['rw'] / max_rw) ** 1.5
        
        # Component 2: Efficiency (Lower min_d is better)
        # Inverse normalize: min_d_min / min_d
        # Non-linear scaling (power 1.5)
        score_eff = (min_min_d / f['min_d']) ** 1.5
        
        # Component 3: Flexibility (Lower n_machines is better)
        # Inverse normalize: min_n_machines / n_machines
        score_flex = min_n_machines / f['n_machines']
        
        total_score = w_rw * score_rw + w_eff * score_eff + w_flex * score_flex
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = f['best_machine']
        elif abs(total_score - best_score) < 1e-9:
            # Tie-breaker: Prefer shorter duration, then fewer remaining ops
            if f['min_d'] < features[best_idx]['min_d']:
                best_idx = i
                best_machine = f['best_machine']
            elif f['min_d'] == features[best_idx]['min_d']:
                if f['remaining_ops'] < features[best_idx]['remaining_ops']:
                    best_idx = i
                    best_machine = f['best_machine']

    # Return (index, machine) to ensure we pick the optimal machine for the chosen op
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END
