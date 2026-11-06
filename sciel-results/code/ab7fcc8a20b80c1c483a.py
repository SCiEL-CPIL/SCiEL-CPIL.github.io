import numpy as np
import time
from collections import defaultdict

def priority(instance):
    start_time = time.time()
    # Generous deadline, computation is tiny
    deadline = start_time + 5.0 
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
                
    # Calculate Critical Path (Longest path from node to sink)
    cp = [0.0] * n
    state = [0] * n # 0: unvisited, 1: in stack, 2: done
    
    def get_cp(i):
        if state[i] == 2:
            return cp[i]
        if state[i] == 1:
            return 0.0 # Cycle protection
            
        state[i] = 1
        max_succ_cp = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                return times[i]
            s_cp = get_cp(j)
            if s_cp > max_succ_cp:
                max_succ_cp = s_cp
        cp[i] = times[i] + max_succ_cp
        state[i] = 2
        return cp[i]
        
    for i in range(n):
        get_cp(i)
        
    # Calculate upstream path (longest path from source to node)
    up = [0.0] * n
    state_up = [0] * n
    
    def get_up(i):
        if state_up[i] == 2:
            return up[i]
        if state_up[i] == 1:
            return 0.0
            
        state_up[i] = 1
        max_pred_up = 0.0
        for pred in predecessors[i]:
            if time.time() > deadline:
                return times[i]
            p_up = get_up(pred)
            if p_up > max_pred_up:
                max_pred_up = p_up
        up[i] = times[i] + max_pred_up
        state_up[i] = 2
        return up[i]
        
    for i in range(n):
        get_up(i)
        
    # Combine heuristics:
    # 1. Critical Path (downstream) - most important
    # 2. Upstream path - helps identify tasks that are bottlenecks from start
    # 3. Task time - tie-breaker
    # 
    # Weighted combination: cp has highest weight, then up, then time
    # This prioritizes tasks that both have long remaining work AND are late in the chain
    priorities = np.asarray(cp, dtype=float) + 0.5 * np.asarray(up, dtype=float) + times * 1e-6
    
    return priorities
