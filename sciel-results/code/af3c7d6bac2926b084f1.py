"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor list and count immediate successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            
    succ_counts = np.zeros(n, dtype=float)
    for i in range(n):
        succ_counts[i] = len(successors[i])
        
    # Calculate bottleneck slack heuristic
    # We want to prioritize tasks with small (cycle_time - time_i) / succ_count
    # This is equivalent to maximizing (succ_count) / (cycle_time - time_i)
    # We add 1 to numerator to give sinks (succ_count=0) some priority based on time
    # We add 1 to denominator to avoid division by zero and smooth the score
    
    epsilon = 1.0
    slack_term = (succ_counts + 1.0) / (cycle_time - times + epsilon)
    
    # Combine with processing time to ensure long tasks are generally prioritized
    # Weighting: The slack term can be large if time is close to cycle_time.
    # Let's normalize or just add them.
    # If cycle_time is 40, and time is 39, slack_term ~ (S+1)/2.
    # If time is 1, slack_term ~ (S+1)/40.
    # Adding time_i (0..40) dominates.
    
    scores = times + 0.1 * slack_term
    
    # Check deadline (though this is fast)
    if time.time() > deadline:
        # Fallback to simple LPT if something went wrong (shouldn't happen)
        return times
        
    return scores
# EVOLVE-BLOCK-END
