import numpy as np
import time

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
        
    # Add small tie-breaker: prefer longer tasks if CP is equal
    # This helps pack large tasks together if they are on similar critical paths
    priorities = np.asarray(cp, dtype=float) + times * 1e-9
    return priorities
