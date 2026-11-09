"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Deadline guard
_START_TIME = None
_DEADLINE = 140.0  # Safety margin before 150s kill

def dispatch(ready_operations, time):
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return first operation or a simple heuristic
        if not ready_operations:
            return 0
        # Simple fallback: Longest Remaining Work
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # If only one operation, return it
    if n_ops == 1:
        return 0
    
    best_idx = 0
    best_score = -float('inf')
    
    for idx in range(n_ops):
        op = ready_operations[idx]
        mo = op.get("machine_options", {})
        
        # Robustly extract min duration from machine_options
        if isinstance(mo, dict) and mo:
            try:
                durations = [v for v in mo.values() if v is not None]
                min_d = min(durations) if durations else 1.0
            except TypeError:
                min_d = 1.0
        else:
            min_d = 1.0
        
        # Calculate Longest Remaining Path (LRP)
        # Check if remaining_ops is a list of dicts
        remaining_ops = op.get("remaining_ops", [])
        
        lrp = 0.0
        if isinstance(remaining_ops, list) and remaining_ops:
            for rem_op in remaining_ops:
                if isinstance(rem_op, dict):
                    rem_mo = rem_op.get("machine_options", {})
                    if isinstance(rem_mo, dict) and rem_mo:
                        try:
                            rem_durations = [v for v in rem_mo.values() if v is not None]
                            lrp += min(rem_durations) if rem_durations else 1.0
                        except TypeError:
                            lrp += 1.0
                    else:
                        lrp += 1.0
                else:
                    lrp += 1.0
        else:
            # Fallback if remaining_ops is not a list of dicts
            # Use remaining_work if available, otherwise assume just current op
            rem_work = op.get("remaining_work", 0)
            if rem_work and isinstance(rem_work, (int, float)):
                lrp = rem_work
            else:
                lrp = min_d
        
        # Ensure LRP is at least the current min duration
        if lrp < min_d:
            lrp = min_d

        # Score: Prioritize jobs with high Critical Path (LRP)
        # Among those, prioritize operations with short processing time (MinDur)
        # This reduces the time the job occupies the system and allows subsequent
        # critical operations to start sooner.
        
        # Add a small constant to denominator to avoid division by zero and smooth
        # the difference between very short and short operations.
        score = lrp / (min_d + 0.1)
        
        if score > best_score:
            best_score = score
            best_idx = idx
        elif abs(score - best_score) < 1e-9:
            # Tie-breaker: prefer shorter min duration
            if isinstance(mo, dict) and mo:
                try:
                    cur_min_d = min(v for v in mo.values() if v is not None)
                except (TypeError, ValueError):
                    cur_min_d = 1.0
                best_mo = ready_operations[best_idx].get("machine_options", {})
                if isinstance(best_mo, dict) and best_mo:
                    try:
                        best_min_d = min(v for v in best_mo.values() if v is not None)
                    except (TypeError, ValueError):
                        best_min_d = 1.0
                else:
                    best_min_d = 1.0
                if cur_min_d < best_min_d:
                    best_idx = idx
    
    return int(best_idx)

# EVOLVE-BLOCK-END
