"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Deadline guard: the evaluator kills after 150s wall-clock.
# We want to return well before that.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 140.0  # safety margin before 150s


def _check_deadline():
    """Return True if we should stop searching."""
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Composite heuristic that balances:
    1. Shortest Processing Time (SPT) on the chosen machine - frees machines quickly
    2. Remaining work of the job - avoids long-tail jobs
    3. Number of remaining operations - prioritize jobs closer to completion
    4. Machine load consideration - prefer less-loaded machines
    
    The dispatch function receives ready_operations (list of dicts) and time.
    Each op dict has: job, op, machine_options {machine: duration}, 
                    remaining_ops, remaining_work
    
    Returns: index of the ready operation to start (int)
    """
    if not ready_operations:
        return 0
    
    # If only one option, return it immediately
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # For each ready operation, compute a composite score
    # We want to pick the operation with the best (lowest) score
    
    # Precompute machine-level statistics from available info
    # Since we don't have global machine state, we use per-op heuristics
    
    best_score = float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        if _check_deadline():
            break
            
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get durations for all machine options
        durations = list(machine_options.values())
        if not durations:
            continue
        
        min_duration = min(durations)
        max_duration = max(durations)
        avg_duration = sum(durations) / len(durations)
        
        # SPT component: prefer shorter processing time (lower is better)
        # Normalize by average to make it comparable
        spt_score = min_duration / (avg_duration + 1e-9)
        
        # Remaining work component: prefer jobs with less remaining work
        # But also consider that jobs with more remaining work might need to start sooner
        # Use a normalized measure
        rw_score = remaining_work / (remaining_work + 1e-9)
        
        # Remaining ops component: prefer jobs with fewer remaining ops (closer to done)
        # This helps with flow shop effects
        ro_score = remaining_ops / (remaining_ops + 1e-9)
        
        # Machine flexibility: prefer operations that have more machine options
        # (more flexibility means less likely to be blocked)
        n_machines = len(machine_options)
        flex_score = 1.0 / (n_machines + 1e-9)
        
        # Variability in processing times: lower variability is generally better
        if len(durations) > 1:
            mean_d = avg_duration
            var_d = sum((d - mean_d) ** 2 for d in durations) / len(durations)
            cv_d = (var_d ** 0.5) / (mean_d + 1e-9)
        else:
            cv_d = 0.0
        
        # Composite score: weighted combination
        # Weights tuned to balance immediate throughput vs. long-term completion
        w_spt = 0.35
        w_rw = 0.25
        w_ro = 0.20
        w_flex = 0.10
        w_cv = 0.10
        
        score = (w_spt * spt_score + 
                 w_rw * rw_score + 
                 w_ro * ro_score + 
                 w_flex * flex_score + 
                 w_cv * cv_d)
        
        if score < best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
