"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Critical Path Method (CPM) Slack.
    Tasks with lower slack (more critical) get higher priority.
    Long tasks among critical ones get a slight boost.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Parse predecessors
    pred = instance["predecessors"]
    
    # Build successor lists
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
            
    # Calculate in-degrees for topological sort
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
        
    # Calculate Earliest Start (ES) times
    # ES[i] = max(ES[p] + times[p]) for all p in pred[i]
    ES = np.zeros(n, dtype=float)
    for node in topo_order:
        max_end = 0.0
        for p in pred[node]:
            if ES[p] + times[p] > max_end:
                max_end = ES[p] + times[p]
        ES[node] = max_end
        
    # Calculate Project Duration
    project_duration = 0.0
    for i in range(n):
        end_time = ES[i] + times[i]
        if end_time > project_duration:
            project_duration = end_time
            
    # Calculate Latest Start (LS) times
    # LS[i] = min(LS[s] - times[i]) for all s in succ[i]
    # For tasks with no successors, LS[i] = project_duration - times[i]
    LS = np.full(n, project_duration, dtype=float)
    
    for node in reversed(topo_order):
        # If node has successors, LS[node] = min(LS[s] for s in succ[node]) - times[node]
        # If node has no successors, LS[node] = project_duration - times[node]
        if succ[node]:
            min_ls_succ = min(LS[s] for s in succ[node])
            LS[node] = min_ls_succ - times[node]
        else:
            LS[node] = project_duration - times[node]
            
    # Calculate Slack
    # Slack = LS - ES
    slack = LS - ES
    
    # Due to floating point or logic, slack might be slightly negative, clamp to 0
    slack = np.maximum(slack, 0.0)
    
    max_slack = slack.max()
    if max_slack == 0:
        # All tasks are critical, prioritize by time
        max_time = times.max()
        if max_time == 0:
            return np.ones(n)
        return times / max_time
        
    # Normalize slack to [0, 1]
    norm_slack = slack / max_slack
    
    # Priority: Inverse of slack (lower slack = higher priority)
    # Add a small component for time to break ties among critical tasks
    max_time = times.max()
    if max_time == 0:
        time_norm = np.zeros(n)
    else:
        time_norm = times / max_time
        
    # We want high priority for low slack.
    # priority = (1 - norm_slack) * weight1 + time_norm * weight2
    # Let's try 0.8 for criticality and 0.2 for time
    priority_scores = 0.8 * (1.0 - norm_slack) + 0.2 * time_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
