"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Combine multiple heuristics:
    1. Shortest Processing Time (SPT) - pick operation with shortest min duration
    2. Most Work Remaining (MWR) - favor jobs with more total work left
    3. Fewest Remaining Operations (FRO) - favor jobs closer to completion
    4. Machine load consideration - prefer machines that are less busy
    
    We use a weighted combination that prioritizes:
    - Shortest processing time on the best available machine
    - Balance between finishing jobs soon and keeping all machines busy
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Calculate scores for each ready operation
    scores = []
    
    # Find min and max values for normalization
    min_durs = [min(op["machine_options"].values()) for op in ready_operations]
    max_durs = [max(op["machine_options"].values()) for op in ready_operations]
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    
    min_min_dur = min(min_durs)
    max_min_dur = max(min_durs)
    min_rem_work = min(remaining_works)
    max_rem_work = max(remaining_works)
    min_rem_ops = min(remaining_ops)
    max_rem_ops = max(remaining_ops)
    
    # Normalize to [0, 1]
    def normalize(val, min_val, max_val):
        if max_val == min_val:
            return 0.5
        return (val - min_val) / (max_val - min_val)
    
    for i, op in enumerate(ready_operations):
        # SPT score: lower duration is better (invert so higher is better)
        spt_score = 1.0 - normalize(min_durs[i], min_min_dur, max_min_dur)
        
        # MWR score: more remaining work is better
        mwr_score = normalize(remaining_works[i], min_rem_work, max_rem_work)
        
        # FRO score: fewer remaining operations is better (invert)
        fro_score = 1.0 - normalize(remaining_ops[i], min_rem_ops, max_rem_ops)
        
        # Weighted combination
        # Emphasize SPT most, then MWR, then FRO
        score = 0.5 * spt_score + 0.3 * mwr_score + 0.2 * fro_score
        
        scores.append(score)
    
    # Return index of operation with highest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, len(scores)):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
