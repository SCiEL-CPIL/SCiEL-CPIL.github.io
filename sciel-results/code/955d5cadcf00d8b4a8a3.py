"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Compute a priority score for each activity in a resource-constrained project scheduling instance.
    
    Strategy: Resource-Weighted Inverse Slack.
    Prioritizes activities that are on the critical path (low total float) AND consume scarce resources.
    This treats the schedule as a flow where critical bottlenecks must be cleared first to prevent
    cascading delays.
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
    # TF = LF - ES - Duration
    TF = LF - ES - durations
    # Ensure TF is non-negative for safety, though it should be >= 0 in CPM
    TF = np.maximum(TF, 0.0)
    
    # Check time budget
    if time.time() - start_time > DEADLINE:
        # Fallback: Return inverse float only
        return 1.0 / (TF + 1e-9)

    # Resource Scarcity Metric
    # For each activity, calculate the "pressure" it exerts on resources.
    # Pressure = sum over resources of (demand / capacity)^2
    # Squaring emphasizes resources that are close to saturation.
    # We normalize by the max possible pressure to keep scales comparable.
    
    # Avoid division by zero
    safe_caps = np.where(capacities > 0, capacities, 1.0)
    ratio = demands / safe_caps
    
    # If capacity is very small relative to demand, the ratio is huge.
    # We cap the ratio to prevent outliers from dominating if data is noisy.
    # However, for RCPSP, high demand/capacity is genuinely critical.
    
    resource_pressure = np.sum(ratio**2, axis=1)
    
    # Normalize resource pressure
    max_pressure = np.max(resource_pressure)
    if max_pressure > 0:
        resource_pressure = resource_pressure / max_pressure
    else:
        resource_pressure = np.ones(n)

    # Criticality Score
    # Inverse Total Float. Activities with 0 float are most critical.
    # Use a soft inverse to avoid infinity.
    eps = 1e-6
    criticality = 1.0 / (TF + eps)
    
    # Normalize criticality?
    # Max criticality is 1/eps which is huge.
    # It's better to scale it relative to the max criticality in the set,
    # or just use the raw inverse float as it naturally orders them.
    # However, combining with resource pressure (0..1) might wash out the criticality
    # if criticality is 1000 and resource is 0.5.
    # Let's normalize criticality to 0..1 range as well, or use a logarithmic scale?
    # Standard MinTF heuristic uses raw TF (lower is better).
    # Here we want Higher Score = Higher Priority.
    # So we want High Criticality (Low TF) and High Pressure.
    
    # Let's normalize Criticality to 0..1
    min_tf = np.min(TF)
    max_tf = np.max(TF)
    if max_tf > min_tf:
        # Map TF [min_tf, max_tf] to [1, 0]
        crit_norm = 1.0 - (TF - min_tf) / (max_tf - min_tf)
    else:
        crit_norm = np.ones(n)
        
    # Combine: 
    # We want to prioritize activities that are BOTH critical AND resource-intensive.
    # Product is a good structural choice for "interaction" effects.
    # Priority = Criticality * Resource Pressure
    
    # To prevent zero priority for non-critical but high resource activities,
    # we can add a small base weight for resource pressure, but typically
    # in RCPSP, non-critical activities should wait.
    
    # Let's try a weighted sum where Criticality is dominant.
    # P = 0.8 * crit_norm + 0.2 * resource_pressure
    
    # Or the product which is more aggressive on bottlenecks:
    # P = crit_norm * resource_pressure
    
    # Let's use a hybrid: 
    # If an activity is critical, its priority is boosted by resource use.
    # If it's not critical, its priority is low, but still influenced by resources.
    
    # Let's go with: Priority = (1 + resource_pressure) * crit_norm
    # This ensures critical activities are always prioritized, but those using
    # more resources get a slight edge among critical ones.
    
    priority_scores = (1.0 + resource_pressure) * crit_norm
    
    # Add a small duration component?
    # Longer activities are harder to fit.
    # max_dur = np.max(durations)
    # dur_norm = durations / max_dur
    # priority_scores = 0.9 * priority_scores + 0.1 * dur_norm
    
    # Final check on time
    if time.time() - start_time > DEADLINE:
         return priority_scores

    return priority_scores
# EVOLVE-BLOCK-END
