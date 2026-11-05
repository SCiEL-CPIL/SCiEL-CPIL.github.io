"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict
import time

def priority(instance):
    """
    Computes priority scores for activities using a hybrid heuristic:
    1. Longest Path (LP) to end (critical path proximity)
    2. Resource contention factor (max demand/capacity ratio)
    
    The final score is a weighted combination that prioritizes activities
    that are both on the critical path AND have high resource demands,
    helping to prevent resource bottlenecks later in the schedule.
    """
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    pred_list = instance.get("predecessors", None)
    if pred_list is None:
        pred_list = [[] for _ in range(n)]
    
    # Get resource data
    # Expected keys: "demands" (n x r matrix) and "capacities" (r,) array
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Build successor list
    succ = defaultdict(list)
    for i in range(n):
        for p in pred_list[i]:
            succ[p].append(i)
    
    # Compute Longest Path (LP) using DFS with memoization
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
        
        lp[u] = durations[u] + max_succ_lp
        state[u] = 2
        return lp[u]
        
    for i in range(n):
        if state[i] == 0:
            dfs(i)
    
    # Compute resource contention factor
    # For each activity, find the maximum (demand/capacity) ratio across all resources
    # This identifies activities that are most likely to cause resource bottlenecks
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        
        # Ensure shapes are correct
        if demands.ndim == 1:
            # Single resource case
            demands = demands.reshape(-1, 1)
            capacities = capacities.reshape(-1)
        
        # Avoid division by zero
        safe_capacities = np.where(capacities > 0, capacities, 1.0)
        
        # Compute ratio for each activity-resource pair
        ratios = demands / safe_capacities  # shape: (n, r)
        
        # For each activity, take the max ratio across resources
        # This represents the "tightest" resource constraint for that activity
        resource_contention = np.max(ratios, axis=1)  # shape: (n,)
    else:
        # No resource data available; use zero contention
        resource_contention = np.zeros(n, dtype=float)
    
    # Normalize LP and resource_contention to [0, 1] range for fair weighting
    lp_min, lp_max = np.min(lp), np.max(lp)
    if lp_max > lp_min:
        lp_norm = (lp - lp_min) / (lp_max - lp_min)
    else:
        lp_norm = np.zeros(n, dtype=float)
    
    rc_min, rc_max = np.min(resource_contention), np.max(resource_contention)
    if rc_max > rc_min:
        rc_norm = (resource_contention - rc_min) / (rc_max - rc_min)
    else:
        rc_norm = np.zeros(n, dtype=float)
    
    # Combine scores: weight LP heavily (0.7) and resource contention (0.3)
    # The LP component ensures critical path activities are scheduled first
    # The resource component ensures high-demand activities are scheduled early
    # to avoid them blocking later activities
    weight_lp = 0.7
    weight_rc = 0.3
    
    scores = weight_lp * lp_norm + weight_rc * rc_norm
    
    # Scale back to original LP scale for consistency with expected score magnitudes
    # This preserves the relative ordering while maintaining a reasonable magnitude
    if lp_max > lp_min:
        scores = lp_min + scores * (lp_max - lp_min)
    else:
        scores = lp
    
    return scores

# EVOLVE-BLOCK-END
