"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = _time.time()


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Uses a composite heuristic combining:
    - Critical Path Method (CPM): prioritize operations on the critical path
    - Minimum Slack Time (MST): prioritize operations with least slack
    - Shortest Processing Time (SPT): break ties by shortest duration
    
    Also includes a deadline guard to ensure we return before the 150s limit.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        # Fallback: pick first ready operation
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # Get machine options
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Remaining work for this job
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # Heuristic components:
        # 1. Criticality: operations with more remaining work are more critical
        #    (they have more work that depends on them or is in parallel)
        criticality = remaining_work / max(1.0, remaining_ops)
        
        # 2. Slack time approximation: 
        #    If an operation has few remaining ops but lots of remaining work,
        #    it might be on a tight path. We approximate slack as:
        #    slack ~ remaining_ops - 1 (fewer remaining ops = less slack)
        slack_factor = max(0.0, (remaining_ops - 1) / max(1.0, remaining_ops))
        
        # 3. Processing time: shorter is generally better to free machines
        #    But very short operations might not be as critical
        spt_score = min_duration / (min_duration + max_duration)  # normalized 0-1
        
        # Composite score:
        # - Primary: higher criticality is better (more remaining work per op)
        # - Secondary: lower slack is better (more urgent)
        # - Tertiary: shorter processing time breaks ties
        
        # We want to maximize: criticality, minimize slack, minimize duration
        # score = criticality - slack_factor - spt_score * 0.5
        
        # Alternative: use a weighted combination that balances urgency and efficiency
        # Weight criticality heavily, then use slack and SPT as tiebreakers
        score = criticality * 1.0 - slack_factor * 0.3 - spt_score * 0.2
        
        # Add a small bonus for operations that are the last in their job
        # (completing a job frees up resources and reduces makespan)
        if remaining_ops == 1:
            score += 0.5
        
        scores[i] = score
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
