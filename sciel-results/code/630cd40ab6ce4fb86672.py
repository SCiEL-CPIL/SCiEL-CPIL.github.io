import numpy as np

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        best_m, best_d = min(op["machine_options"].items(), key=lambda x: x[1])
        return (0, best_m)
    
    scores = np.zeros(n)
    best_machines = [None] * n
    
    # Precompute normalization constants
    max_rem_work = max(op["remaining_work"] for op in ready_operations)
    if max_rem_work == 0:
        max_rem_work = 1.0
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores[i] = rem_work
            best_machines[i] = None
            continue
            
        # Find machine with min duration and count options
        min_dur = float('inf')
        best_m = None
        n_options = 0
        for m, d in machine_options.items():
            n_options += 1
            if d < min_dur:
                min_dur = d
                best_m = m
        
        # Flexibility: fewer options = more constrained = higher priority
        # Use inverse: 1/n_options as constraint factor
        flexibility = 1.0 / n_options if n_options > 0 else 0.0
        
        # Normalized remaining work (0 to 1)
        norm_work = rem_work / max_rem_work
        
        # Heuristic Score:
        # - High remaining work: critical path length
        # - Low min duration: quick completion frees up resources
        # - Fewer machine options: more constrained, schedule earlier
        # - More remaining ops: more work left, prioritize
        score = 2.0 * norm_work + 1.5 * flexibility - 0.3 * (min_dur / max_rem_work) + 0.2 * rem_ops
        
        scores[i] = score
        best_machines[i] = best_m
        
    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
