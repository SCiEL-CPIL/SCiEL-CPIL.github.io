import numpy as np
import time

def priority(instance):
    """
    Calculates priority scores for assembly tasks to minimize the number of stations.
    Uses a composite heuristic: Downstream Subtree Weight + Branching Unlock Bonus.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    times = np.array(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency list for successors
    # predecessors[i] is a list of tasks that must come before i
    # We need successors: for task j, which tasks i have j in predecessors[i]?
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Topological sort (Kahn's algorithm) to get processing order
    # We need reverse topological order to compute downstream weights (sinks first)
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1 # Wait, in_degree for topological sort usually counts incoming edges.
            # Here, predecessors[i] are the incoming edges to i.
            # So in_degree[i] = len(predecessors[i])
            
    # Recalculate in_degree correctly
    in_degree = [len(p) for p in predecessors]
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    
    # Simple queue for topological sort
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order length < n, there's a cycle. But task says "Simple Assembly Line", assume acyclic.
    # If cycle, fallback to LPT
    if len(topo_order) < n:
        return times.copy()

    # Compute Downstream Weight (W_down)
    # W_down[u] = times[u] + sum(W_down[v] for v in successors[u])
    W_down = np.zeros(n, dtype=float)
    
    # Process in reverse topological order (sinks to sources)
    for u in reversed(topo_order):
        w = times[u]
        for v in successors[u]:
            w += W_down[v]
        W_down[u] = w
        
    # Check deadline
    if time.time() > deadline:
        return W_down

    # Compute Out-degree (number of immediate successors)
    out_degree = np.array([len(s) for s in successors], dtype=float)
    
    # Calculate Priority
    # Base: Downstream Weight
    # Bonus: Branching Unlock (Out-degree) scaled by task time relative to cycle time
    # The "tightness" or "bottleneck" aspect is captured by t_i / cycle_time.
    # A task that consumes a large fraction of the cycle time and unlocks many branches
    # is a high-priority candidate to place early to maximize parallelism in remaining capacity.
    
    # Let's use a small weight for the bonus to not overwhelm the structural weight.
    # W_down is sum of times. Max W_down is sum of all times.
    # Bonus term: (t_i / cycle_time) * out_degree[i]
    # This term is roughly O(1) to O(3).
    # W_down is roughly O(10) to O(100).
    # So the bonus is a small perturbation.
    
    bonus = (times / cycle_time) * out_degree
    
    # We can also add a small tie-breaker based on task duration itself (LPT)
    # to prefer longer tasks if weights are identical.
    # But W_down already includes times.
    
    # Let's try: P = W_down + 0.5 * bonus
    # The factor 0.5 is a heuristic guess for "light boost".
    
    scores = W_down + 0.5 * bonus
    
    # Ensure no negative scores (shouldn't happen)
    scores = np.maximum(scores, 0)
    
    return scores
