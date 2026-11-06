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
    # 3. Bottleneck slack - favor tasks that consume most station capacity
    #    relative to branching complexity they unlock
    # 4. Task time - tie-breaker
    
    # Compute bottleneck slack heuristic:
    # For each task i, compute ratio of (cycle_time - time_i) to number of immediate successors
    # Tasks with small ratio consume more available capacity per successor unlocked
    # This favors "bottleneck" tasks that should be placed early to maximize packing
    cycle_time = float(instance.get("cycle_time", 40.0))
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Avoid division by zero: tasks with no successors get a high slack (low priority)
    # Use max(1, num_successors) to handle terminal tasks
    denom = np.maximum(num_successors, 1.0)
    slack = (cycle_time - times) / denom
    
    # Invert slack so that smaller slack (higher priority) gets larger score
    # Use negative slack to make it a "priority" (higher = better)
    # Add a small constant to avoid issues with very large negative values
    bottleneck_score = -slack
    
    # Weighted combination:
    # cp has highest weight (longest remaining path downstream)
    # up has moderate weight (longest path from source)
    # bottleneck_score favors tasks that are capacity bottlenecks
    # times is a tiny tie-breaker
    priorities = (np.asarray(cp, dtype=float) 
                  + 0.5 * np.asarray(up, dtype=float) 
                  + 0.3 * bottleneck_score 
                  + times * 1e-6)
    
    return priorities
