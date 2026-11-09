"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using a Critical Path Method (CPM) based Slack heuristic.
    Tasks with lower slack (less flexibility in scheduling) are given higher priority.
    This is a distinct approach from simple positional weight sums.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    pred = instance["predecessors"]
    
    # Build successor lists
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_copy = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
                
    if len(topo_order) != n:
        # Fallback for cycles
        return times.copy()
        
    # Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    
    for node in topo_order:
        if pred[node]:
            es[node] = max(ef[p] for p in pred[node])
        else:
            es[node] = 0.0
        ef[node] = es[node] + times[node]
        
    # Project End Time (Max EF)
    project_end = max(ef)
    
    # Backward Pass: Calculate Latest Finish (LF) and Latest Start (LS)
    lf = np.zeros(n, dtype=float)
    ls = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        if succ[node]:
            lf[node] = min(ls[s] for s in succ[node])
        else:
            lf[node] = project_end
        ls[node] = lf[node] - times[node]
        
    # Calculate Slack
    slack = ls - es
    
    # Priority: Lower Slack = Higher Priority
    # We invert slack and normalize.
    # To handle potential negative slacks (though unlikely in valid CPM without fixed deadline),
    # we shift them.
    min_slack = slack.min()
    max_slack = slack.max()
    slack_range = max_slack - min_slack
    
    if slack_range > 0:
        # Normalize slack to [0, 1], then invert so lower slack -> higher priority
        norm_slack = (slack - min_slack) / slack_range
        priority_scores = 1.0 - norm_slack
    else:
        # All tasks have same slack
        priority_scores = np.ones(n, dtype=float)
        
    # Add a small boost for longer tasks to break ties effectively
    max_time = times.max() if times.max() > 0 else 1.0
    time_norm = times / max_time
    priority_scores = 0.95 * priority_scores + 0.05 * time_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
