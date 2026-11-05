"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    """
    Computes priority scores for activities based on the Longest Path (LP) heuristic.
    LP[i] = duration[i] + max(LP[j]) for all successors j of i.
    This prioritizes activities on or near the critical path.
    """
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors. The interface says 'predecessors'.
    # Assume format: list of lists, where instance['predecessors'][i] is a list of indices of preds of i.
    pred_list = instance.get("predecessors", None)
    
    if pred_list is None:
        # Fallback: if no predecessors provided, assume no constraints (just LDF)
        # Or maybe the key is different? The prompt says "instance has ... predecessors".
        # Let's assume standard key. If missing, default to empty.
        pred_list = [[] for _ in range(n)]
    
    # Build successor list for reverse graph traversal
    # succ[j] contains all activities i such that j is a predecessor of i
    succ = defaultdict(list)
    in_degree = np.zeros(n, dtype=int)
    
    for i in range(n):
        for p in pred_list[i]:
            # p is predecessor of i, so i is successor of p
            succ[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    # We need to process nodes in topological order to calculate LP.
    # Actually, for LP (Longest Path from node to end), we can process in reverse topological order.
    # Or, we can do a DFS with memoization.
    # Given n=30, DFS is very safe and simple.
    
    # Let's use DFS with memoization for simplicity and robustness against small graph quirks.
    # State: 0 = unvisited, 1 = visiting (cycle detection, though RCPSP should be DAG), 2 = visited
    state = np.zeros(n, dtype=int)
    lp = np.zeros(n, dtype=float)
    
    def dfs(u):
        if state[u] == 2:
            return lp[u]
        if state[u] == 1:
            # Cycle detected. In a valid RCPSP this shouldn't happen.
            # Return 0 or handle gracefully.
            return 0.0
        state[u] = 1
        
        max_succ_lp = 0.0
        # Iterate over successors of u
        # Note: succ is built from pred_list. 
        # Wait, my succ construction above:
        # for i in range(n): for p in pred_list[i]: succ[p].append(i)
        # So succ[p] are the activities that depend on p.
        # Yes, these are the successors of p.
        
        for v in succ[u]:
            val = dfs(v)
            if val > max_succ_lp:
                max_succ_lp = val
        
        lp[u] = durations[u] + max_succ_lp
        state[u] = 2
        return lp[u]
        
    for i in range(n):
        if state[i] == 0:
            dfs(i)
            
    return lp

# EVOLVE-BLOCK-END
