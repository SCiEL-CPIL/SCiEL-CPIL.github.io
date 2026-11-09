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
    neg_vals = [-v for v in values]
    max_neg = max(neg_vals)
    if max_neg > 500:
        return min(values)
    exp_sum = sum(math.exp((nv - max_neg) / temperature) for nv in neg_vals)
    soft_min_val = -(max_neg + math.log(exp_sum)) * temperature
    return max(0.1, soft_min_val)

def _select_machine(mo, temperature=0.3):
    """Select machine using softmax weighting over negative durations."""
    if not mo:
        return None
    if len(mo) == 1:
        return list(mo.keys())[0]

    machines = list(mo.keys())
    durations = [mo[m] for m in machines]

    neg = [-d for d in durations]
    max_neg = max(neg)
    if max_neg > 500:
        return min(mo, key=mo.get)

    exps = [math.exp((nv - max_neg) / temperature) for nv in neg]
    total = sum(exps)
    if total <= 0:
        return min(mo, key=mo.get)

    best_i = max(range(len(machines)), key=lambda i: exps[i])
    best_w = exps[best_i]

    second_best_w = 0.0
    for i in range(len(machines)):
        if i != best_i and exps[i] > second_best_w:
            second_best_w = exps[i]

    if second_best_w > 0.95 * best_w:
        best_i = min(range(len(machines)), key=lambda i: durations[i])

    return machines[best_i]

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
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
            best_machine = _select_machine(mo, temperature=0.4)
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
            best_machine = _select_machine(mo, temperature=0.4)
            sm = _soft_min(durations, temperature=0.3)
            if n_machines > 1:
                sorted_durs = sorted(durations)
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
        
    if var_factor > 1.0:
        var_factor = 1.0
    elif var_factor < 0.0:
        var_factor = 0.0

    # Weights:
    w_rw = 0.45 + 0.25 * var_factor
    w_eff = 0.45 - 0.15 * var_factor
    w_flex = 0.08
    w_ops = 0.09
    
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
        
        if f['rw'] == 0:
            score_rw = 0.0
        else:
            score_rw = (f['rw'] / max_rw) ** 1.2
        
        score_eff = (min_min_d / f['soft_min_d']) ** 1.3
        
        score_flex = min_n_machines / f['n_machines']
        
        if max_remaining_ops == 0:
            score_ops = 0.0
        else:
            score_ops = f['remaining_ops'] / max_remaining_ops
        
        score_flex_ratio = 1.0 / (1.0 + max(0.0, f['flex_ratio'] - 1.0))
        
        total_score = w_rw * score_rw + w_eff * score_eff + w_flex * score_flex + w_ops * score_ops + 0.05 * score_flex_ratio
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = f['best_machine']
        elif abs(total_score - best_score) < 1e-9:
            if f['min_d'] < features[best_idx]['min_d']:
                best_idx = i
                best_machine = f['best_machine']
            elif abs(f['min_d'] - features[best_idx]['min_d']) < 1e-9:
                if f['flex_ratio'] > features[best_idx]['flex_ratio']:
                    best_idx = i
                    best_machine = f['best_machine']

    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
