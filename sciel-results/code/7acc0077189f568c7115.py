"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR (Most Work Remaining)
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        if n == 1:
            return 0
        
        # Strategy: Critical Path Prioritization with Load Balancing
        # 1. Estimate the "Critical Path" length for each ready operation.
        #    Since we don't have the full DAG, we approximate the critical path
        #    contribution of an operation as its duration plus the remaining work
        #    of the job it belongs to.
        #    However, a better proxy for "urgency" in flexible job shop is often
        #    the Longest Remaining Path (LRP).
        #    LRP_j = max over all paths in remaining DAG.
        #    We only have 'remaining_work' (sum of durations).
        #    Let's assume 'remaining_work' is a good proxy for LRP.
        #    
        # 2. Machine Selection:
        #    If we return (index, machine), we can choose the machine.
        #    The prompt says "optionally (index, machine)".
        #    To improve makespan, we should choose the machine that minimizes the
        #    completion time of the critical job, or balances load.
        #    Since we don't know machine loads, we assume the simulator handles
        #    machine assignment if we return just index.
        #    BUT, if we can return (index, machine), we have more control.
        #    Let's check if returning a tuple is supported and beneficial.
        #    The interface says: "dispatch(ready_operations, time) -> index ... (optionally (index, machine))"
        #    This implies the evaluator can accept a tuple.
        #    If we return a tuple, we must ensure the machine is in machine_options.
        #    
        #    Let's try returning just the index first, but with a better heuristic.
        #    The previous attempt used a composite score.
        #    Let's try a "Weighted Critical Path" approach.
        #    
        #    Score for operation i:
        #    - Higher remaining_work is better (more critical).
        #    - Lower min_duration is better (faster to finish, less blocking).
        #    
        #    Let's try: Score = remaining_work / (min_duration + 1)
        #    This prioritizes jobs with lots of work left but short current ops.
        
        best_idx = 0
        best_score = -1e10
        
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            if not mo:
                min_dur = 1.0
            else:
                min_dur = min(mo.values())
            
            # Heuristic: Prioritize high remaining work, penalize long duration
            # This is similar to "Longest Processing Time" for critical jobs,
            # but "Shortest Processing Time" for non-critical.
            # A common effective heuristic is "Critical Ratio" or "Weighted SPT".
            # Let's use: Score = rw * (1 / (min_dur + 1))
            score = rw / (min_dur + 1.0)
            
            if score > best_score + 1e-9:
                best_score = score
                best_idx = i
            elif abs(score - best_score) <= 1e-9:
                # Tie breaker: pick the one with higher remaining work
                if rw > ready_operations[best_idx].get("remaining_work", 0):
                    best_idx = i
        
        return int(best_idx)

    except Exception:
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
