import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Strategy: Min Slack with Shortest Processing Time (SPT) Bias.
    
    The current program uses Min Slack + Resource Pressure (High Demand).
    This strategy uses Min Slack + Shortest Duration (SPT).
    
    Rationale:
    - Min Slack identifies activities on the critical path (must not be delayed).
    - SPT bias prioritizes shorter activities among those with similar slack.
    - Short activities complete quickly, releasing resources sooner for subsequent 
      activities, which is beneficial in resource-constrained environments where 
      long activities hold resources for extended periods.
    - This differs from the current "Resource Pressure" approach which favors 
      heavy users, potentially locking up resources for a long time if the 
      heavy user is also long.
    """
    start_time = time.time()
    
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback if graph is invalid or cyclic (shouldn't happen)
        return -durations.copy()

    # Forward Pass: Earliest Start (ES) and Earliest Finish (EF)
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan = max(EF)
    
    # Backward Pass: Latest Finish (LF) and Latest Start (LS)
    LF = [makespan] * n
    LS = [makespan] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan
        LS[j] = LF[j] - durations[j]
        
    # Calculate Slack (Total Float)
    Slack = [LS[j] - ES[j] for j in range(n)]
    
    # Strategy: Min Slack (High Priority) + Min Duration (High Priority)
    # We want high score = high priority.
    # Low Slack is good -> -Slack is high.
    # Short Duration is good -> -Duration is high.
    
    # Normalize Slack and Duration to comparable scales?
    # Slack is in time units. Duration is in time units.
    # They are directly comparable.
    
    # Weighting:
    # Slack is the primary driver (critical path).
    # Duration is a secondary driver to break ties or fine-tune.
    # If we use -Slack - 0.1 * Duration, it's similar to current but inverted duration sign.
    # Current: -Slack + Pressure.
    # New: -Slack - Duration.
    
    # Let's try a stronger SPT bias to make it materially different.
    # If an activity has Slack 0, and another has Slack 0, the shorter one goes first.
    # If an activity has Slack 1, and another has Slack 0.5, the Slack 0.5 one goes first 
    # unless the duration difference is huge.
    
    # Let's use a coefficient of 1.0 for Duration to make it significant.
    # Score = -Slack - 1.0 * Duration
    
    scores = np.array([-s - d for s, d in zip(Slack, durations)])
    
    return scores
