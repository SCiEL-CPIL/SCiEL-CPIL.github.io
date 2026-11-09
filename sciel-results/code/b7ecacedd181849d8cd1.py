"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each assembly task for the station-filling algorithm.
    Uses Positional Weight (total downstream work) as the primary signal,
    with immediate successor count as a tie-breaker.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
        
    # Build successor list and in-degree count
    succ = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        plist = preds[i] if isinstance(preds[i], (list, tuple, np.ndarray)) else []
        if isinstance(plist, (list, tuple, np.ndarray)):
            for p in plist:
                p_int = int(p)
                if 0 <= p_int < n:
                    succ[p_int].append(i)
                    in_degree[i] += 1
    
    # Topological sort (Kahn's algorithm)
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
                
    # Handle potential cycles or disconnected components (shouldn't happen in valid ALB)
    if len(topo_order) < n:
        # Fallback to simple LPT if graph is invalid
        return times.copy()

    # Calculate Positional Weights (PW)
    # PW[i] = time[i] + sum(PW[s] for s in succ[i])
    # Process in reverse topological order
    pw = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        total_succ_pw = 0.0
        for s in succ[node]:
            total_succ_pw += pw[s]
        pw[node] = times[node] + total_succ_pw
        
    # Calculate Immediate Successor Count
    is_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Calculate Total Successor Count (optional, for stronger tie-breaking)
    # But IS is usually sufficient and faster.
    # Let's add a small weight to IS to break ties favoring tasks that unlock more work.
    
    # Final Score: PW is dominant. IS breaks ties.
    # We multiply IS by a small factor so it doesn't overpower PW.
    # If PW values are close, IS decides.
    score = pw + 0.1 * is_count
    
    # Optional: Add a very small factor of task time to break ties between tasks
    # with same PW and same IS?
    # score += 0.01 * times
    
    return score
# EVOLVE-BLOCK-END
