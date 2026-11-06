import numpy as np
from collections import defaultdict

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Calculate immediate successor counts
    succ_counts = np.zeros(n, dtype=float)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            succ_counts[p] += 1
            
    # Calculate slack
    slack = cycle_time - times
    
    # Heuristic: Prioritize tasks with smallest ratio of Slack to Immediate Successors.
    # Priority is inversely proportional to (Slack / SuccCount).
    # So Priority = SuccCount / Slack.
    # To avoid division by zero and handle 0 successors:
    # If SuccCount is 0, the task is a sink. 
    # The heuristic implies sinks have "infinite" ratio (if slack>0) or 0/0.
    # A common variation is to use (SuccCount + 1) / (Slack + 1) or similar to keep them in the mix.
    # However, strict interpretation: 
    # We want to maximize Priority.
    # Priority = 1 / (Slack / (SuccCount + epsilon)) = (SuccCount + epsilon) / Slack.
    # Let's use a small epsilon for slack to avoid inf.
    # And for SuccCount, if it's 0, we might want to give it a low but non-zero priority, 
    # or treat it as 1 to not zero it out completely?
    # Let's try: Score = (succ_counts + 1.0) / (slack + 1.0)
    # This ensures sinks (succ=0) have score 1/(slack+1).
    # Tasks with many successors and low slack get high scores.
    
    epsilon = 1e-6
    scores = (succ_counts + 1.0) / (slack + epsilon)
    
    return scores
