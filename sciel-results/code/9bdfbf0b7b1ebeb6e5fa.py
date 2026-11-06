"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute a priority score for each activity in a resource-constrained project scheduling instance.
    
    Strategy: Raw Criticality Boosted by Resource Pressure.
    Uses the inverse of Total Float (from CPM) as the primary priority signal, 
    multiplied by a factor that increases with resource contention.
    This ensures that activities on the critical path are always prioritized over 
    non-critical ones, while among critical activities, those consuming scarcer 
    resources are scheduled first to prevent resource-induced delays.
    """
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin for 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors and build successors
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    # Ensure predecessors are lists of ints
    preds = []
    for i in range(n):
        p = predecessors[i]
        if p is None:
            preds.append([])
        else:
            preds.append(list(p))
            
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            successors[i].append(j)
    
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None:
        demands = np.ones((n, 1))
    else:
        demands = np.asarray(demands, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
    
    if capacities is None:
        capacities = np.ones(demands.shape[1])
    else:
        capacities = np.asarray(capacities, dtype=float)
        if capacities.ndim == 0:
            capacities = np.array([capacities])

    # Handle edge case: no resources or zero capacities
    if demands.shape[1] == 0 or capacities.size == 0 or np.any(capacities == 0):
        # Fallback to duration-based if resources are invalid
        return durations / (np.max(durations) + 1e-9)

    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)])
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    # If cycle detected (should not happen in valid RCPSP), fallback
    if len(topo_order) < n:
        topo_order = list(range(n))

    # Forward Pass: Earliest Start (ES)
    ES = np.zeros(n, dtype=float)
    for i in topo_order:
        for pred in preds[i]:
            if ES[pred] + durations[pred] > ES[i]:
                ES[i] = ES[pred] + durations[pred]
        
    # Project End Time (Makespan of CPM)
    makespan_cpm = np.max(ES + durations)
    
    # Backward Pass: Latest Finish (LF)
    LF = np.full(n, makespan_cpm, dtype=float)
    for i in reversed(topo_order):
        for succ in successors[i]:
            if LF[succ] - durations[succ] < LF[i]:
                LF[i] = LF[succ] - durations[succ]
                
    # Total Float (Slack)
    TF = LF - ES - durations
    # Ensure TF is non-negative for safety
    TF = np.maximum(TF, 0.0)
    
    # Check time budget
    if time.time() - start_time > DEADLINE:
        # Fallback: Return inverse float only
        return 1.0 / (TF + 1e-9)

    # Resource Pressure Metric
    # Pressure = sum over resources of (demand / capacity)^2
    # This highlights activities that are close to saturating a resource.
    safe_caps = np.where(capacities > 0, capacities, 1.0)
    ratio = demands / safe_caps
    resource_pressure = np.sum(ratio**2, axis=1)
    
    # Normalize resource pressure to [0, 1]
    max_pressure = np.max(resource_pressure)
    if max_pressure > 0:
        resource_pressure = resource_pressure / max_pressure
    else:
        resource_pressure = np.zeros(n)

    # Criticality Score
    # Use raw inverse float to maintain strict ordering based on criticality.
    # Activities with 0 float have huge priority.
    eps = 1e-6
    criticality = 1.0 / (TF + eps)
    
    # Combine:
    # Priority = Criticality * (1 + Resource Pressure)
    # This ensures:
    # 1. Critical activities (high criticality) always beat non-critical ones.
    # 2. Among critical activities, those with higher resource pressure are prioritized.
    # 3. Non-critical activities get a small boost from resource pressure, 
    #    but remain far below critical activities due to the 1/TF term.
    
    priority_scores = criticality * (1.0 + resource_pressure)
    
    # Final check on time
    if time.time() - start_time > DEADLINE:
        return priority_scores

    return priority_scores
# EVOLVE-BLOCK-END
