import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = 140.0

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        # Fallback: Simple WSPT
        best_idx = 0
        best_score = -float('inf')
        for i in range(n):
            rw = ready_operations[i]["remaining_work"]
            mds = list(ready_operations[i]["machine_options"].values())
            md = min(mds) if mds else 0
            score = rw / max(md, 1e-6)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    # Calculate features for each operation
    # 1. Soft-min duration
    # 2. Remaining Work
    # 3. Remaining Ops (Depth)
    # 4. Avg Work (RW / Depth)
    
    features = []
    for i, op in enumerate(ready_operations):
        durations = list(op["machine_options"].values())
        if not durations:
            d_soft = 0.0
        else:
            # Soft min using log-sum-exp trick for stability
            # min(d) approx -1/k * log(sum(exp(-k*d)))
            # k=10 is a reasonable smoothness parameter
            k = 10.0
            max_d = max(durations)
            # Shift by max_d to prevent overflow in exp
            # exp(-k * (d - max_d)) is safe
            # Actually, for min, we want to emphasize the smallest.
            # log-sum-exp for min: -1/k * log(sum(exp(-k * d)))
            # To avoid underflow/overflow, we can just use the raw values if they are small.
            # Durations are usually small integers or floats.
            sum_exp = 0.0
            for d in durations:
                sum_exp += math.exp(-k * d)
            if sum_exp == 0:
                d_soft = 0.0
            else:
                d_soft = -1.0 / k * math.log(sum_exp)
                
        rw = op["remaining_work"]
        depth = op["remaining_ops"]
        if depth == 0:
            depth = 1 # Safety
        avg_work = rw / depth
        
        features.append({
            'rw': rw,
            'depth': depth,
            'd': d_soft,
            'avg': avg_work
        })
    
    # Normalize features to [0, 1]
    def normalize(vals):
        if not vals:
            return [0.0] * len(vals)
        min_v = min(vals)
        max_v = max(vals)
        range_v = max_v - min_v
        if range_v < 1e-9:
            return [0.5] * len(vals) # All same, neutral
        return [(v - min_v) / range_v for v in vals]
    
    rws = [f['rw'] for f in features]
    depths = [f['depth'] for f in features]
    ds = [f['d'] for f in features]
    avgs = [f['avg'] for f in features]
    
    n_rws = normalize(rws)
    n_depths = normalize(depths)
    n_ds = normalize(ds)
    n_avgs = normalize(avgs)
    
    # Weights from cell finding
    w_rw = 1.2
    w_depth = 1.8
    w_d = -0.6
    w_avg = 0.4
    
    best_idx = 0
    best_score = -float('inf')
    
    for i in range(n):
        score = (w_rw * n_rws[i] + 
                 w_depth * n_depths[i] + 
                 w_d * n_ds[i] + 
                 w_avg * n_avgs[i])
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
