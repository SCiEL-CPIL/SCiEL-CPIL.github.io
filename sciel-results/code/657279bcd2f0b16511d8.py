import numpy as np
import time as time_module

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    Returns the index of the ready operation to start next.
    
    Strategy: Hybrid MWR + LPT heuristic.
    - Primary: Remaining Work (MWR) to ensure critical path jobs progress.
    - Secondary: Maximum Minimum Duration (LPT-like) to balance loads and prevent late bottlenecks.
    - Tertiary: Remaining Ops count to prioritize job completion.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    start_t = time_module.perf_counter()
    DEADLINE = 145.0

    # Precompute features for all ready operations
    scores = np.zeros(n_ops, dtype=float)
    
    # Store features for tie-breaking and secondary scoring
    rws = np.zeros(n_ops, dtype=float)
    min_durs = np.zeros(n_ops, dtype=float)
    max_durs = np.zeros(n_ops, dtype=float)
    ops_counts = np.zeros(n_ops, dtype=float)
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        remaining_ops = op.get("remaining_ops", 0.0)
        mo = op.get("machine_options", {})
        
        if not mo:
            scores[i] = -np.inf
            continue
        
        durations = np.array(list(mo.values()), dtype=float)
        min_dur = np.min(durations)
        max_dur = np.max(durations)
        
        # Soft-min surrogate for robustness against noise in single-machine options
        # Using a small temperature to stay close to min but smooth
        # soft_min = -log(sum(exp(-d/t))) * t
        # For small t, approx min.
        # We stick to min_dur for speed, but adjust weights.
        
        rws[i] = rw
        min_durs[i] = min_dur
        max_durs[i] = max_dur
        ops_counts[i] = remaining_ops
        
        # Base score: Remaining Work (MWR)
        # This is the strongest predictor for makespan in open shop.
        score = rw
        
        # LPT Component: Prioritize operations with longer *minimum* durations.
        # Increased weight from 0.05 to 0.1 to more aggressively schedule inflexible long ops.
        score += 0.1 * min_dur
        
        # Risk Penalty: Penalize operations with very high maximum durations.
        # If the best machine is busy, this op might be forced onto a slow machine.
        # Small weight to not override MWR.
        score -= 0.01 * max_dur
        
        # Completion Bonus: Jobs with fewer remaining operations are closer to finishing.
        # Finishing a job removes it from the "active" set, potentially simplifying future decisions.
        # This term is small to not override the main heuristics.
        if remaining_ops > 0:
            score += 0.02 / remaining_ops
            
        scores[i] = score
    
    # Check deadline
    if time_module.perf_counter() - start_t > DEADLINE:
        # Fallback: Simple MWR
        return int(np.argmax(rws))
    
    # Find the best index
    best_idx = int(np.argmax(scores))
    
    # Tie-breaking:
    # If scores are very close, we need a robust tie-breaker.
    # 1. Prefer higher remaining work (safety).
    # 2. Prefer shorter min duration (SPT) to clear quick tasks if work is equal? 
    #    Actually, our main score already boosts long min_durs. 
    #    If we are tied in the composite score, it means the trade-off is balanced.
    #    Let's break ties by preferring the operation that finishes sooner (min_dur)
    #    to keep the machine free for the next decision, UNLESS the remaining work is significantly higher.
    #    However, since we already added 0.1 * min_dur to the score, a tie implies
    #    that one op has higher RW but lower MinDur, and the other has lower RW but higher MinDur.
    #    In this specific "balanced" case, standard practice often favors the one with higher RW
    #    to ensure the critical path is handled.
    
    max_score = scores[best_idx]
    # Use a small epsilon for floating point comparison
    ties = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-9]
    
    if len(ties) > 1:
        # Tie-breaker logic:
        # 1. Max Remaining Work (Most Critical)
        # 2. Min Remaining Ops (Closest to Completion)
        # 3. Min Duration (Fastest to clear)
        
        def tiebreak_key(idx):
            return (-rws[idx], ops_counts[idx], min_durs[idx])
        
        best_idx = min(ties, key=tiebreak_key)
        
    return best_idx
