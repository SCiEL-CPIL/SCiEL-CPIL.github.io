"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    """
    Computes priority scores using resource-aware longest path.
    Score[i] = adjusted_duration[i] + max(Score[j]) for all successors j of i,
    where adjusted_duration accounts for resource scarcity.
    Higher score = higher priority.
    """
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    pred_list = instance.get("predecessors", None)
    if pred_list is None:
        pred_list = [[] for _ in range(n)]
    
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Build successor list
    succ = defaultdict(list)
    for i in range(n):
        for p in pred_list[i]:
            succ[p].append(i)
    
    # Compute resource scarcity factor for each activity
    # If resource r has capacity C_r and activity i demands D_ir,
    # the scarcity ratio is D_ir / C_r. Higher ratio = more scarce.
    # We use the max ratio across resources to represent time expansion.
    
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        
        # demands shape: (n, num_resources) or (n,) for single resource
        # capacities shape: (num_resources,) or scalar
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
        
        if np.isscalar(capacities) or (hasattr(capacities, 'ndim') and capacities.ndim == 0):
            capacities = np.array([float(capacities)])
        else:
            capacities = np.asarray(capacities, dtype=float)
        
        # Compute scarcity ratio: for each activity, max over resources of demand/capacity
        # Handle zero capacity (shouldn't happen in valid instances)
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        ratios = demands / safe_cap  # shape: (n, num_resources)
        scarcity_factor = np.max(ratios, axis=1)  # shape: (n,)
        
        # Adjusted duration: duration * (1 + scarcity_factor)
        # This expands duration for activities on scarce resources
        adjusted_durations = durations * (1.0 + scarcity_factor)
    else:
        adjusted_durations = durations.copy()
    
    # DFS with memoization for resource-aware longest path
    state = np.zeros(n, dtype=int)
    lp = np.zeros(n, dtype=float)
    
    def dfs(u):
        if state[u] == 2:
            return lp[u]
        if state[u] == 1:
            return 0.0
        state[u] = 1
        
        max_succ_lp = 0.0
        for v in succ[u]:
            val = dfs(v)
            if val > max_succ_lp:
                max_succ_lp = val
        
        lp[u] = adjusted_durations[u] + max_succ_lp
        state[u] = 2
        return lp[u]
        
    for i in range(n):
        if state[i] == 0:
            dfs(i)
            
    return lp

# EVOLVE-BLOCK-END
