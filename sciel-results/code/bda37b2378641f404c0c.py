"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Score each activity using the Minimum Total Slack heuristic.
    
    This approach calculates the Total Slack (LS - ES) for each activity.
    Activities with lower slack have less flexibility in the schedule and
    are therefore prioritized to be scheduled first. This is a standard
    heuristic for RCPSP that accounts for both forward and backward
    dependencies, often outperforming simple critical path length heuristics.
    """
    n = len(instance["durations"])
    if n == 0:
        return np.array([])
        
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
                
    # Calculate in-degrees for topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                in_degree[j] += 1
                
    # Topological Sort (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    # Handle potential cycles (should not occur in valid RCPSP)
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Forward Pass: Calculate Earliest Start (ES) and Earliest Finish (EF)
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    
    for node in topo_order:
        if predecessors[node]:
            max_pred_ef = 0.0
            for pred in predecessors[node]:
                if pred is not None:
                    if EF[pred] > max_pred_ef:
                        max_pred_ef = EF[pred]
            ES[node] = max_pred_ef
        else:
            ES[node] = 0.0
        EF[node] = ES[node] + durations[node]
        
    # Determine Project End Time
    project_end = np.max(EF) if n > 0 else 0.0
    
    # Backward Pass: Calculate Latest Finish (LF) and Latest Start (LS)
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        if not successors[node]:
            LF[node] = project_end
        else:
            min_succ_ls = float('inf')
            for succ in successors[node]:
                if LS[succ] < min_succ_ls:
                    min_succ_ls = LS[succ]
            LF[node] = min_succ_ls
            
        LS[node] = LF[node] - durations[node]
        
    # Calculate Total Slack
    slack = LS - ES
    # Ensure slack is non-negative
    slack = np.maximum(slack, 0.0)
    
    # Priority: Higher score = Higher Priority.
    # Lower slack indicates higher criticality.
    # We invert slack so that lower slack yields a higher score.
    max_slack = np.max(slack)
    scores = max_slack - slack
    
    return scores
# EVOLVE-BLOCK-END
