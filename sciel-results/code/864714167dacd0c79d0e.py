import time as _time

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
    max_rem_ops = 0.0
    min_min_d = float('inf')
    max_min_d = 0.0
    min_n_machines = float('inf')
    
    features = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        rem_ops = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
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
            'rem_ops': rem_ops,
            'min_d': min_d,
            'n_machines': n_machines,
            'best_machine': best_machine
        })
        
        if rw > max_rw:
            max_rw = rw
        if rem_ops > max_rem_ops:
            max_rem_ops = rem_ops
        if min_d < min_min_d:
            min_min_d = min_d
        if min_d > max_min_d:
            max_min_d = min_d
        if n_machines < min_n_machines:
            min_n_machines = n_machines

    # Handle edge cases for normalization
    if max_rw == 0: max_rw = 1.0
    if max_rem_ops == 0: max_rem_ops = 1.0
    if min_min_d == 0: min_min_d = 1.0
    if max_min_d == 0: max_min_d = 1.0
    if min_n_machines == 0: min_n_machines = 1.0

    best_score = -1.0
    best_idx = 0
    best_machine = None

    for i in range(n):
        f = features[i]
        
        # Component 1: Remaining Work (Higher is better)
        score_rw = f['rw'] / max_rw
        
        # Component 2: Remaining Operations (Higher is better - more steps left)
        score_mor = f['rem_ops'] / max_rem_ops
        
        # Component 3: Efficiency (Lower min_d is better)
        # Inverse normalize: min_d_min / min_d
        score_eff = min_min_d / f['min_d']
        
        # Component 4: Flexibility (Lower n_machines is better)
        # Inverse normalize: min_n_machines / n_machines
        score_flex = min_n_machines / f['n_machines']
        
        # Weights
        # Prioritize criticality (Work + Steps) over efficiency/flexibility
        w_rw = 0.45
        w_mor = 0.35
        w_eff = 0.15
        w_flex = 0.05
        
        total_score = w_rw * score_rw + w_mor * score_mor + w_eff * score_eff + w_flex * score_flex
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            best_machine = f['best_machine']
        elif abs(total_score - best_score) < 1e-9:
            # Tie-breaker: Prefer shorter duration, then fewer machines
            if f['min_d'] < features[best_idx]['min_d']:
                best_idx = i
                best_machine = f['best_machine']
            elif f['min_d'] == features[best_idx]['min_d'] and f['n_machines'] < features[best_idx]['n_machines']:
                best_idx = i
                best_machine = f['best_machine']

    # Return (index, machine) to ensure we pick the optimal machine for the chosen op
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
