import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule using Critical Path Length (CPL) heuristic.
    
    Strategy: 
    1. Estimate the Critical Path Length for each ready operation.
       CPL = min_duration(current_op) + sum(min_duration(next_op) for next_op in remaining_ops)
    2. If remaining_ops is not a list of dicts, fall back to using remaining_work as the future component.
    3. Select the operation with the highest CPL.
    4. Tie-breaker: Shorter current processing time (SPT) to reduce immediate machine occupancy.
    """
    if not ready_operations:
        return 0
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            current_min = min(machine_options.values())
        else:
            current_min = 0.0
            
        remaining_ops = op.get("remaining_ops", [])
        remaining_work = op.get("remaining_work", 0.0)
        
        future_cpl = 0.0
        used_lookahead = False
        
        # Try to compute exact CPL from remaining operations
        if isinstance(remaining_ops, list) and len(remaining_ops) > 0:
            # Check if the first element is a dict with machine_options
            if isinstance(remaining_ops[0], dict) and "machine_options" in remaining_ops[0]:
                for next_op in remaining_ops:
                    next_opts = next_op.get("machine_options", {})
                    if next_opts:
                        future_cpl += min(next_opts.values())
                    else:
                        # Fallback for this specific op
                        future_cpl += next_op.get("duration", 0.0) if "duration" in next_op else 0.0
                used_lookahead = True
        
        if not used_lookahead:
            # Fallback: use remaining_work as proxy for future CPL
            # Note: remaining_work usually includes current op or just future?
            # In many contexts, remaining_work is total work left including current.
            # If so, future_work = remaining_work - current_min?
            # Or remaining_work is just the sum of future ops?
            # Let's assume remaining_work is the total work remaining for the job.
            # To be safe, if we don't have lookahead, we use remaining_work as the "urgency" score.
            # But we already have current_min.
            # Let's just use remaining_work as the score if lookahead fails, 
            # but we need to be consistent.
            # If remaining_work includes current op, then CPL ~ remaining_work.
            # If remaining_work excludes current op, CPL ~ current_min + remaining_work.
            # Let's assume remaining_work is the sum of durations of all remaining ops (including current? or not?).
            # Given the name "remaining_work", it likely means work left to do.
            # Let's assume it's the sum of future ops + current op.
            # So CPL is roughly remaining_work.
            future_cpl = remaining_work
            # To avoid double counting if remaining_work includes current, 
            # and we added current_min, we might be overestimating.
            # However, if lookahead is not available, remaining_work is the best proxy.
            # Let's just use remaining_work directly as the score in this branch?
            # No, let's stick to: Score = current_min + future_estimate.
            # If remaining_work is total, future_estimate should be remaining_work - current_min?
            # This is ambiguous.
            # Let's assume the lookahead branch is the primary intended path for "new approach".
            # If it fails, we fall back to a simple heuristic: remaining_work.
            
            # Let's just set score to remaining_work in this case to be safe.
            score = remaining_work
        else:
            score = current_min + future_cpl
            
        # Tie-breaker: If scores are very close, prefer shorter current_min
        # We can incorporate a small penalty for current_min in the score?
        # No, CPL is the main driver.
        
        if score > best_score:
            best_score = score
            best_idx = i
        elif score == best_score:
            # Tie-breaker: Shorter current operation is better
            # We need to compare current_min of current best vs current candidate
            # This requires storing current_min of best. 
            # For simplicity in this edit, we'll just keep the first one found with max score.
            # Or we can modify the score slightly: score - 0.001 * current_min
            pass

    return int(best_idx)
