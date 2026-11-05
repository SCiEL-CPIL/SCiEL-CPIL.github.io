"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Critical Ratio (CR) heuristic.
    
    For each ready operation, we estimate the slack time for its job as:
        slack = (sum of remaining processing times for all unfinished ops in the job) - (time until some deadline)
    
    However, since we don't have explicit deadlines, we use a proxy:
    The critical ratio is typically: CR = (due date - current time) / remaining processing time
    
    Since we don't have due dates, we adapt the concept:
    - For each job, the "remaining work" is the sum of all remaining processing times.
    - We consider the urgency based on how much work remains relative to the current time context.
    
    A better interpretation for job-shop without explicit due dates:
    - Prioritize jobs with less remaining work (they are closer to completion).
    - But also consider machine load: if a machine is busy, it might be better to pick an op that can go on a less loaded machine.
    
    Let's use a combined heuristic:
    1. For each ready operation, compute the total remaining processing time for its job (sum of min durations across all remaining ops).
    2. The "slack" proxy: jobs with less remaining work are more urgent to finish.
    3. Additionally, prefer operations that can be processed on a machine with lower estimated load.
    
    Actually, let's stick closer to the CR idea:
    - Estimate the "due date" for each job as current_time + some multiple of remaining work? That's circular.
    
    Alternative: Use the ratio of remaining_work to the number of remaining_ops. Jobs with fewer remaining ops but still significant work might be more critical.
    
    Let me think differently. In job-shop scheduling, common heuristics include:
    - Shortest Processing Time (SPT)
    - Longest Processing Time (LPT)
    - Minimum Slack Time
    - Critical Ratio
    
    Since we don't have due dates, let's use:
    - For each ready operation, compute the minimum possible duration across its machine options.
    - Prioritize operations whose job has the least remaining work (to finish jobs quickly).
    - Among ties, prefer operations with shorter processing times.
    
    Actually, let's implement a proper CR-like heuristic:
    - For each job, remaining_work is given.
    - We can estimate the "urgency" as remaining_work / remaining_ops (average work per remaining op).
    - Jobs with higher average work per op might be more critical if they have complex operations.
    
    Let's try: prioritize by (remaining_work / max(1, remaining_ops)) in ascending order? 
    No, that would prioritize jobs with smaller average work.
    
    Actually, for makespan minimization, we want to avoid idle time and balance load.
    A good heuristic is to pick the operation that:
    1. Belongs to a job with the most remaining work (to keep busy jobs busy) - this is what the current program does.
    2. Can be processed on a machine that is likely to be free soon.
    
    Since we don't have machine availability info, let's use:
    - For each ready operation, find the minimum duration across its machine options.
    - Prioritize operations with the smallest minimum duration (SPT-like).
    - But also consider the job's remaining work.
    
    Combined score: score = remaining_work * min_duration
    Pick the operation with the highest score? Or lowest?
    
    Let's try: prioritize operations from jobs with high remaining work, but among those, pick the one with the shortest processing time to reduce makespan.
    
    Actually, let's implement a simple but effective heuristic:
    - For each ready operation, compute the minimum processing time across its machine options.
    - Compute a priority score: score = remaining_work / (min_duration + epsilon)
    - Pick the operation with the highest score.
    
    This balances: jobs with more remaining work get higher priority, but operations with shorter processing times also get boosted.
    """
    if not ready_operations:
        return 0
    
    scores = []
    for op in ready_operations:
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Get the minimum duration across all machine options
        machine_options = op["machine_options"]
        if machine_options:
            min_duration = min(machine_options.values())
        else:
            min_duration = 1.0
        
        # Avoid division by zero
        min_duration = max(min_duration, 0.1)
        
        # Critical ratio-like score: higher means more urgent
        # We want to prioritize jobs with more remaining work, but also favor shorter operations
        score = remaining_work / min_duration
        scores.append(score)
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    return best_idx
# EVOLVE-BLOCK-END
