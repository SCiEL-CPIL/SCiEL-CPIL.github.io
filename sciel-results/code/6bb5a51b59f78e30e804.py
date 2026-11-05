import numpy as np
from collections import defaultdict

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Parse predecessors
    preds = instance["predecessors"]
    # preds is likely a list of lists, where preds[i] is a list of indices
    # Ensure it's a list of lists
    if isinstance(preds, np.ndarray):
        # If it's a 2D array, convert to list of lists
        preds = [list(row) for row in preds]
    elif isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], (list, np.ndarray)):
        preds = [list(row) for row in preds]
    else:
        # Maybe it's a flat list or something else? Assuming standard format: list of lists
        preds = [list(row) for row in preds]
        
    # Build successors
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            
    # Forward Pass: ES, EF
    ES = np.zeros(n)
    EF = np.zeros(n)
    
    # Topological sort or simply iterate until no changes (since n is small)
    # Or use a simple DFS/BFS if we want to be safe, but for 30 nodes, iterative relaxation is fine.
    # Let's do a proper topological sort to be safe and fast.
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Use a stack or list, order doesn't strictly matter for ES/EF if we check all preds
        u = queue.pop(0) 
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If graph is not a DAG or has issues, topo_order might be incomplete. 
    # But RCPSP instances are DAGs.
    if len(topo_order) != n:
        # Fallback to LPT if topo sort fails
        return durations

    for u in topo_order:
        if preds[u]:
            ES[u] = max(EF[p] for p in preds[u])
        else:
            ES[u] = 0
        EF[u] = ES[u] + durations[u]
        
    T = max(EF)
    
    # Backward Pass: LF, LS
    LF = np.full(n, T)
    LS = np.full(n, T)
    
    # Reverse topo order
    for u in reversed(topo_order):
        if succs[u]:
            LF[u] = min(LS[v] for v in succs[u])
        else:
            LF[u] = T
        LS[u] = LF[u] - durations[u]
        
    slack = LS - ES
    
    # Priority: Lower slack is better (higher priority).
    # Multiply by a large factor to make slack the primary key.
    # Add duration as a secondary key (higher duration is better).
    # Since we want highest priority value to be selected:
    # High Priority = Low Slack.
    # So Priority = -Slack * K + Duration.
    # If Slack is 0, Priority = Duration.
    # If Slack is 10, Priority = -10*K + Duration.
    # We want Slack=0 to be higher than Slack=10.
    # -0*K + D1 > -10*K + D2  =>  D1 > D2 - 10*K.
    # If K is large enough (e.g., > max_duration), this holds.
    max_dur = max(durations) if n > 0 else 1
    K = max_dur * 2 + 1
    
    priorities = -slack * K + durations
    
    return priorities
