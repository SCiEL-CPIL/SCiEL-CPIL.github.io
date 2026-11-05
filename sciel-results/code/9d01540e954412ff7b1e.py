"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute priority scores using a Resource-Weighted Critical Path heuristic.
    Score = Critical Path Length * (1 + Resource Scarcity Factor).
    This prioritizes activities that are both structurally critical and resource-intensive.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    
    # Get resources and capacities
    # Assume instance has 'demands' (n x R) and 'capacities' (R,)
    # If not present, fallback to static CP
    if "demands" not in instance or "capacities" not in instance:
        return _static_cp_priority(instance)
        
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Handle zero capacities to avoid division by zero
    # If capacity is 0, demand must be 0 or it's infeasible. Assume feasible.
    # For scarcity ratio, if cap is 0 and demand > 0, treat as infinite priority?
    # Or just use a large number.
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity = np.where(capacities > 0, demands / capacities, np.where(demands > 0, 1e6, 0))
    
    # Sum scarcity across resources for each activity
    # This represents the "resource load" of the activity
    resource_load = np.sum(scarcity, axis=1)
    
    # Compute Static Critical Path Length (Longest Path to End)
    cp_length = _compute_cp_length(n, durations, predecessors)
    
    # If CP computation failed (cycle), fallback
    if cp_length is None:
        return durations.copy()
        
    # Combine: Weight the CP length by the resource load
    # We want activities that are critical AND resource-heavy to have high priority.
    # A simple multiplicative model:
    # Priority = CP * (1 + alpha * ResourceLoad)
    # Let's try alpha = 1.0 for now.
    # Or maybe additive? CP + ResourceLoad?
    # Multiplicative emphasizes the intersection.
    
    # Normalize resource load to avoid dominance if scales are very different
    # But CP and ResourceLoad are both roughly O(n) and O(R).
    
    # Let's use: Priority = CP * (1 + ResourceLoad)
    # This ensures that if ResourceLoad is 0, it falls back to CP.
    # If ResourceLoad is high, it boosts the CP.
    
    priority_scores = cp_length * (1.0 + resource_load)
    
    # Check time
    if time.time() > deadline:
        return cp_length # Fallback to static CP if calculation took too long (unlikely)
        
    return priority_scores

def _compute_cp_length(n, durations, predecessors):
    """Compute longest path from each node to the end."""
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != n:
        return None
        
    # Longest path to end
    cp = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            cp[node] = durations[node]
        else:
            max_succ = max(cp[s] for s in successors[node])
            cp[node] = durations[node] + max_succ
            
    return cp

def _static_cp_priority(instance):
    """Fallback to static CP if resources are not provided."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    cp = _compute_cp_length(n, durations, predecessors)
    if cp is None:
        return durations.copy()
    return cp
# EVOLVE-BLOCK-END
