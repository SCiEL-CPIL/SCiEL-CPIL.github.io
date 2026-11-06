import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Multi-factor heuristic for open-shop:
    - MWR (Most Work Remaining) as primary criterion
    - Prioritize jobs closer to completion (fewer remaining ops)
    - Prefer shorter processing times to reduce makespan
    - Balance machine loads by preferring less-loaded machines
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    start_t = time_module.perf_counter()
    DEADLINE = 100.0

    # Compute aggregate statistics for normalization
    total_rw = 0.0
    min_remaining_ops = float('inf')
    max_remaining_ops = 0
    all_min_durs = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        if mo:
            durs = list(mo.values())
            if durs:
                all_min_durs.append(min(durs))
        total_rw += rw
        if ro < min_remaining_ops:
            min_remaining_ops = ro
        if ro > max_remaining_ops:
            max_remaining_ops = ro
    
    if total_rw == 0:
        total_rw = 1.0
    if min_remaining_ops == float('inf'):
        min_remaining_ops = 1
    if max_remaining_ops <= min_remaining_ops:
        max_remaining_ops = min_remaining_ops + 1
    
    min_min_dur = min(all_min_durs) if all_min_durs else 1.0
    max_min_dur = max(all_min_durs) if all_min_durs else 1.0
    if max_min_dur <= min_min_dur:
        max_min_dur = min_min_dur + 1.0

    scores = []
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})
        
        if not mo:
            scores.append(0.0)
            continue
        
        durations = list(mo.values())
        min_dur = min(durations) if durations else 1.0
        avg_dur = sum(durations) / len(durations)
        n_machines = len(mo)
        
        # Normalize remaining work to [0, 1]
        rw_norm = rw / total_rw
        
        # Normalize remaining ops: fewer ops = higher score (closer to completion)
        # Invert: jobs with fewer remaining ops get higher priority
        if remaining_ops > 0:
            ops_norm = 1.0 - (remaining_ops - min_remaining_ops) / (max_remaining_ops - min_remaining_ops)
        else:
            ops_norm = 1.0
        
        # Normalize min duration: shorter = higher score
        min_dur_norm = 1.0 - (min_dur - min_min_dur) / (max_min_dur - min_min_dur)
        
        # Composite score: weighted combination
        # MWR is primary, then completion proximity, then short duration
        score = 0.6 * rw_norm + 0.3 * ops_norm + 0.1 * min_dur_norm
        
        scores.append(score)
    
    # Check deadline
    if time_module.perf_counter() - start_t > DEADLINE:
        best_rw = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        return best_idx
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking: if scores are very close, use secondary criteria
    max_score = scores[best_idx]
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-6]
    
    if len(ties) > 1:
        def tiebreak_score(idx):
            op = ready_operations[idx]
            rw = op.get("remaining_work", 0)
            remaining_ops = op.get("remaining_ops", 0)
            mo = op.get("machine_options", {})
            durations = list(mo.values()) if mo else [1.0]
            min_dur = min(durations) if durations else 1.0
            
            # Lower is better: fewer remaining ops, shorter duration
            return (remaining_ops, min_dur, -rw)
        
        best_idx = min(ties, key=tiebreak_score)
        
    return best_idx
