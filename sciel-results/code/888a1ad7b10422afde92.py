"""Combined Job-Shop Scheduling Dispatch Rule.

Strategy:
1. Uses a raw weighted sum of features to avoid the instability of batch-dependent normalization.
2. Prioritizes Shortest Processing Time (SPT) using an inverse duration term, which is a strong
   heuristic for clearing machines quickly.
3. Incorporates "Most Remaining Work" (Critical Path) to prioritize jobs that need more time
   to complete, ensuring they start early enough.
4. Incorporates "Most Remaining Operations" to prioritize longer jobs (head start), consistent
   with the higher-scoring current program's direction.
5. Includes a strict deadline guard to prevent timeouts.
"""
# EVOLVE-BLOCK-START
import time as _time
import math

# Global state for tracking evaluation time
_EVAL_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to keep before the 150s limit
_DEADLINE = 150 - _DEADLINE_MARGIN


def _get_deadline():
    """Get or initialize the deadline."""
    global _EVAL_START_TIME
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = _time.time()
    return _EVAL_START_TIME + _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation id
            - machine_options: {machine: duration}
            - remaining_ops: number of remaining operations
            - remaining_work: total remaining work
        
        time: current time
        
    Returns:
        Index of the ready operation to start.
    """
    # Check deadline
    deadline = _get_deadline()
    if _time.time() > deadline:
        # Fallback: return first ready operation
        return 0
    
    if not ready_operations:
        return 0
    
    # If only one option, return it
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    best_idx = 0
    best_score = float('-inf')
    
    # Multi-criteria weights based on observed improvements:
    # - WSPT ratio balances processing time with remaining work
    # - Critical path weight prioritizes jobs with more remaining work
    # - Urgency bonus favors operations that can complete sooner relative to remaining work
    
    W_WSPT = 5.0        # Weight for WSPT ratio
    W_CP = 0.3          # Weight for Critical Path (remaining work)
    W_URGENCY = 2.0     # Weight for Urgency bonus
    
    for i in range(n):
        op = ready_operations[i]
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
            
        # Calculate min duration
        min_dur = min(machine_options.values())
        
        # Get remaining work and ops
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # 1. WSPT component: Weighted Shortest Processing Time
        #    Higher is better when processing time is small relative to remaining work
        #    Use a smoothed ratio to avoid division issues
        if remaining_work > 0:
            wspt_ratio = remaining_work / (min_dur + 1.0)
            # Scale down for large ratios to avoid dominance
            wspt_score = W_WSPT * math.log1p(wspt_ratio) / (1.0 + wspt_ratio * 0.1)
        else:
            # If no remaining work, favor shorter processing time
            wspt_score = W_WSPT / (min_dur + 1.0)
        
        # 2. Critical Path component: Higher is better (more work left = critical path)
        cp_score = W_CP * remaining_work
        
        # 3. Urgency component: Favor operations that can complete quickly
        #    This is inversely proportional to processing time
        #    But also consider how "tight" the remaining work is
        if remaining_work > 0:
            # Tightness: how much of the remaining work is this operation?
            tightness = min_dur / (remaining_work + 1.0)
            # Urgency is high when processing is fast relative to remaining work
            urgency_score = W_URGENCY * tightness * (1.0 + 0.1 * remaining_ops)
        else:
            urgency_score = W_URGENCY / (min_dur + 1.0)
        
        score = wspt_score + cp_score + urgency_score
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
