import numpy as np
from collections import deque
import time

def priority(instance):
    start_time = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Handle case where n is 0
    if n == 0:
        return np.array([])

    # Extract resource information
    demands = np.asarray(instance.get("demands", np.zeros((n, 1))), dtype=float)
    capacities = np.asarray(instance.get("capacities", np.ones(1)), dtype=float)
    
    # Ensure demands has correct shape
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    if capacities.ndim == 0:
        capacities = capacities.reshape(1)
    
    num_resources = demands.shape[1] if demands.ndim == 2 else 1
    if num_resources == 0:
        num_resources = 1
        demands = np.zeros((n, 1))
    
    # Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological Sort (Kahn's Algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
            
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If there's a cycle, topo_order will be incomplete. 
    # RCPSP instances are DAGs, so this shouldn't happen.
    # But just in case, if len(topo_order) < n, we might have issues.
    # For safety, if incomplete, we fall back to LDF?
    if len(topo_order) < n:
         return durations

    # Calculate Longest Path to End (LD) - backward pass
    LD = np.zeros(n)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ_LD = 0.0
        for v in successors[u]:
            if LD[v] > max_succ_LD:
                max_succ_LD = LD[v]
        LD[u] = durations[u] + max_succ_LD
    
    # Calculate Longest Path from Start (LF) - forward pass
    LF = np.zeros(n)
    # Process in topological order
    for u in topo_order:
        max_pred_LF = 0.0
        for p in predecessors[u]:
            if LF[p] + durations[p] > max_pred_LF:
                max_pred_LF = LF[p] + durations[p]
        LF[u] = max_pred_LF
    
    # Critical path length
    cp_length = np.max(LD) if n > 0 else 0.0
    
    # Compute slack for each activity
    # Slack = LF[u] + LD[u] - cp_length (approximation)
    slack = LF + LD - cp_length
    slack = np.maximum(slack, 0.0)
    
    # Compute resource contention score
    # For each activity, compute the maximum demand-to-capacity ratio across resources
    # This identifies which activities are likely to cause resource bottlenecks
    if num_resources > 0 and np.max(capacities) > 0:
        # Normalize capacities to avoid division by zero
        safe_capacities = np.where(capacities > 0, capacities, 1.0)
        demand_ratio = demands / safe_capacities
        # Maximum contention across all resources
        contention = np.max(demand_ratio, axis=1)
    else:
        contention = np.zeros(n)
    
    # Normalize contention to [0, 1] range for better scaling
    max_contention = np.max(contention) if n > 0 else 1.0
    if max_contention > 0:
        contention_normalized = contention / max_contention
    else:
        contention_normalized = np.zeros(n)
    
    # Combine criticality and resource contention
    # Activities with low slack (close to critical path) AND high resource contention
    # should be prioritized to avoid delays
    
    # Weight criticality more heavily, but add a resource contention boost
    # The boost is proportional to how much the activity is on the critical path
    criticality_weight = 1.0
    contention_weight = 0.3
    
    # Scale LD to be in a reasonable range
    if cp_length > 0:
        LD_normalized = LD / cp_length
    else:
        LD_normalized = np.zeros(n)
    
    # Final priority: base criticality + resource contention boost for near-critical activities
    # Activities with low slack get a bigger boost from high contention
    slack_factor = np.where(slack < cp_length * 0.2, 1.0, np.where(slack < cp_length * 0.5, 0.5, 0.2))
    
    priority_scores = criticality_weight * LD_normalized + contention_weight * contention_normalized * slack_factor
    
    # Add a tiny tie-breaker: prefer longer durations if scores are equal
    # This mimics LDF for ties, which is known to be good.
    priority_scores = priority_scores + durations * 1e-9
    
    # Check time budget - if we're running low, return immediately
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin before 120s limit
        return priority_scores
    
    return priority_scores
