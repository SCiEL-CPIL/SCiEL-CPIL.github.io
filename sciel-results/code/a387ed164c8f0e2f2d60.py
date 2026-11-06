import numpy as np
import time as _time

_START_TIME = None
_DEADLINE = 115.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    if not ready_operations:
        return 0
    
    # Deadline guard: if we are approaching the limit, fall back to a fast heuristic
    if _time.time() - _START_TIME > _DEADLINE:
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0
    
    try:
        # Compute machine load estimates from the ready queue
        machine_load = {}
        for op in ready_operations:
            for m, d in op["machine_options"].items():
                if m not in machine_load:
                    machine_load[m] = 0
                machine_load[m] += d
        
        scores = []
        for op in ready_operations:
            durations = list(op["machine_options"].values())
            if not durations:
                scores.append(0.0)
                continue
            
            min_dur = min(durations)
            sum_dur = sum(durations)
            remaining_work = op["remaining_work"]
            remaining_ops = op["remaining_ops"]
            num_machines = len(op["machine_options"])
            
            # Handle zero duration
            if min_dur <= 0:
                min_dur = 1e-6
            
            # Primary signal: Critical Path Slack
            # remaining_work / min_duration favors jobs with short initial steps relative to total work
            slack = remaining_work / min_dur
            
            # Secondary signal: Efficiency factor from current program
            # Favors operations that can be completed quickly on the best machine
            efficiency = sum_dur / min_dur
            
            # Machine contention: prefer machines with lower current load
            min_load = min(machine_load[m] for m in op["machine_options"])
            
            # Flexibility-based contention avoidance
            # Fewer machine options means higher contention risk
            flexibility_penalty = 0.05 * (num_machines - 1) if num_machines > 1 else 0.0
            
            # Composite score:
            # - Primary: slack (higher is better) - short steps relative to remaining work
            # - Secondary: efficiency (higher is better) - quick completion on best machine
            # - Tertiary: remaining work (higher is better) - prioritize jobs with more work
            # - Penalty: machine load and low flexibility
            score = 1.0 * slack + 0.3 * efficiency + 0.1 * remaining_work - 0.02 * min_load - flexibility_penalty
            
            scores.append(score)
        
        return int(np.argmax(scores))
    except Exception:
        return 0
