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
    
    scores = np.zeros(n)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            scores[i] = rem_work
            best_machines[i] = None
            continue
            
        # Find machine with min duration
        min_dur = float('inf')
        best_m = None
        for m, d in machine_options.items():
            if d < min_dur:
                min_dur = d
                best_m = m
        
        # Heuristic Score
        # Prioritize high remaining work, low duration
        # Normalize? 
        # Let's use a simple weighted sum
        score = 1.0 * rem_work - 0.5 * min_dur + 0.1 * rem_ops
        
        scores[i] = score
        best_machines[i] = best_m
        
    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
