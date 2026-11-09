import numpy as np

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick best machine for the single op
        best_m, best_d = min(op["machine_options"].items(), key=lambda x: x[1])
        return (0, best_m)
    
    # Compute features for all operations
    rem_works = np.array([op["remaining_work"] for op in ready_operations])
    rem_ops = np.array([op["remaining_ops"] for op in ready_operations])
    min_durs = np.zeros(n)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        
        if not machine_options:
            min_durs[i] = float('inf')
            best_machines[i] = None
            continue
            
        # Find machine with min duration
        min_dur = float('inf')
        best_m = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_m = m
        
        min_durs[i] = min_dur
        best_machines[i] = best_m
    
    # Replace inf with a large value for numerical stability
    min_durs = np.where(np.isinf(min_durs), 1e6, min_durs)
    
    # Normalize features to [0, 1] range for better weighting
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_work = normalize(rem_works)
    norm_dur = normalize(min_durs)
    norm_ops = normalize(rem_ops)
    
    # Composite urgency score:
    # - High remaining work is urgent (prioritize)
    # - Low duration is good (can finish quickly)
    # - More remaining ops means more work to do (prioritize)
    # Weighting based on open-shop scheduling heuristics
    scores = 0.5 * norm_work - 0.3 * norm_dur + 0.2 * norm_ops
    
    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
