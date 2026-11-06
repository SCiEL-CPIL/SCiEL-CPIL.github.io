import numpy as np
import time
from collections import defaultdict

def priority(instance):
    start_time = time.time()
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
                
    # Calculate Total Downstream Work (TDW)
    # TDW[i] = time[i] + sum(TDW[j] for j in successors[i])
    # This is the sum of times of i and all transitive successors.
    tdw = [0.0] * n
    state = [0] * n # 0: unvisited, 1: in stack, 2: done
    
    def get_tdw(i):
        if state[i] == 2:
            return tdw[i]
        if state[i] == 1:
            return 0.0 # Cycle protection
            
        state[i] = 1
        sum_succ_tdw = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                return times[i]
            s_tdw = get_tdw(j)
            sum_succ_tdw += s_tdw
        tdw[i] = times[i] + sum_succ_tdw
        state[i] = 2
        return tdw[i]
        
    for i in range(n):
        get_tdw(i)
        
    # Calculate Longest Successor Path (LSP) - same as current cp
    lsp = [0.0] * n
    state_lsp = [0] * n
    
    def get_lsp(i):
        if state_lsp[i] == 2:
            return lsp[i]
        if state_lsp[i] == 1:
            return 0.0
            
        state_lsp[i] = 1
        max_succ_lsp = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                return times[i]
            s_lsp = get_lsp(j)
            if s_lsp > max_succ_lsp:
                max_succ_lsp = s_lsp
        lsp[i] = times[i] + max_succ_lsp
        state_lsp[i] = 2
        return lsp[i]
        
    for i in range(n):
        get_lsp(i)
        
    # Calculate Number of Successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Heuristic:
    # TDW is the primary driver: tasks that "support" more total work should go first.
    # LSP is a secondary driver: tasks on the critical path should go first.
    # Num Successors is a tertiary driver: tasks that unlock many branches should go first.
    
    priorities = (np.asarray(tdw, dtype=float) 
                  + 0.5 * np.asarray(lsp, dtype=float) 
                  + 0.1 * num_successors 
                  + times * 1e-6)
    
    return priorities
