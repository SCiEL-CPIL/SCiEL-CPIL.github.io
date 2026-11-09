"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 150.0

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        # Compute features
        max_rw = max(op.get("remaining_work", 0) for op in ready_operations)
        if max_rw == 0:
            max_rw = 1.0
        
        # Pre-compute min durations
        min_durations = []
        for op in ready_operations:
            mo = op.get("machine_options", {})
            if mo:
                min_durations.append(min(mo.values()))
            else:
                min_durations.append(1.0)
        
        # Find max min_duration for normalization
        max_min_d = max(min_durations) if min_durations else 1.0
        if max_min_d == 0:
            max_min_d = 1.0
        
        best_idx = 0
        best_score = -1.0
        
        # Use a composite score that balances remaining work and efficiency
        # We want to maximize remaining work (urgency) and minimize processing time (efficiency)
        # Using inverse scaling for efficiency to keep scores positive and comparable
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            min_d = min_durations[i]
            
            # Work score: higher is better
            work_score = rw / max_rw
            
            # Efficiency score: we want shorter durations to have higher scores
            # Using 1/(min_d + epsilon) to avoid division by zero and keep values bounded
            # Adding a small offset to prevent extreme values when min_d is very small
            eff_score = 1.0 / (min_d + 0.1)
            
            # Normalize efficiency score to [0, 1] range for better combination
            # The maximum possible eff_score is 1/(0.1) = 10
            # So we scale it down
            eff_score_norm = eff_score / 10.0
            
            # Composite score: weight work more heavily as it's often more critical for makespan
            # But also consider efficiency to avoid long operations blocking machines
            score = 0.7 * work_score + 0.3 * eff_score_norm
            
            # Secondary tie-breaker: prefer operations with fewer remaining steps (to complete jobs faster)
            # This can help reduce the number of active jobs and free up machines
            remaining_ops = op.get("remaining_ops", 0)
            tie_breaker = -remaining_ops * 0.01  # Small penalty for more remaining ops
            
            total_score = score + tie_breaker
            
            if total_score > best_score:
                best_score = total_score
                best_idx = i
            elif abs(total_score - best_score) < 1e-9:
                # Final tie-breaker: prefer shorter min duration
                if min_d < min_durations[best_idx]:
                    best_idx = i
        
        return int(best_idx)

    except Exception:
        # Robust fallback
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
