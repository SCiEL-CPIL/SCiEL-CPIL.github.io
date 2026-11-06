"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque


def priority(instance):
    """
    Score each activity based on a resource-bottleneck heuristic combining:
      1. Resource scarcity: sum over resources of (demand / capacity)
      2. Critical path length: longest path from this activity to the end
      3. Predecessor count (indirect measure of how "ready" or constrained)
    
    Higher score = higher priority.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    
    # Build successor list from predecessors
    successors = defaultdict(list)
    pred_counts = np.zeros(n, dtype=int)
    if predecessors is not None:
        for i, preds in enumerate(predecessors):
            pred_counts[i] = len(preds)
            for p in preds:
                successors[p].append(i)
    
    # Compute critical path length (longest path to end) using topological order
    # First, get topological order
    in_degree = pred_counts.copy()
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # If topo_order doesn't include all nodes (cycle), fall back to original order
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Longest path to end (reverse topological order)
    longest_to_end = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if successors[node]:
            longest_to_end[node] = durations[node] + max(
                longest_to_end[s] for s in successors[node]
            )
        else:
            longest_to_end[node] = durations[node]
    
    # Resource scarcity score
    if demands is not None and capacities is not None:
        # demands could be a list of lists or a 2D array
        if isinstance(demands, (list, np.ndarray)):
            demands_arr = np.asarray(demands, dtype=float)
            capacities_arr = np.asarray(capacities, dtype=float)
            
            # Handle different shapes
            if demands_arr.ndim == 1:
                # Single resource
                with np.errstate(divide='ignore', invalid='ignore'):
                    scarcity = np.where(capacities_arr > 0, 
                                        demands_arr / np.maximum(capacities_arr, 1e-10), 
                                        0)
            elif demands_arr.ndim == 2:
                # Multiple resources: sum of demand/capacity ratios
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(capacities_arr > 0,
                                      demands_arr / np.maximum(capacities_arr, 1e-10),
                                      0)
                scarcity = np.sum(ratios, axis=1)
            else:
                scarcity = np.zeros(n)
        else:
            scarcity = np.zeros(n)
    else:
        scarcity = np.zeros(n)
    
    # Normalize scores to comparable scales
    max_dur = np.max(durations) if n > 0 else 1.0
    max_lte = np.max(longest_to_end) if n > 0 else 1.0
    max_scarcity = np.max(scarcity) if n > 0 and np.any(scarcity > 0) else 1.0
    
    # Combine: weight critical path more heavily, then resource scarcity
    # Add a small component for duration itself
    if max_lte > 0:
        lte_norm = longest_to_end / max_lte
    else:
        lte_norm = np.zeros(n)
    
    if max_scarcity > 0:
        scarcity_norm = scarcity / max_scarcity
    else:
        scarcity_norm = np.zeros(n)
    
    if max_dur > 0:
        dur_norm = durations / max_dur
    else:
        dur_norm = np.zeros(n)
    
    # Weighted combination
    # Critical path is most important, then scarcity, then duration
    scores = 0.5 * lte_norm + 0.35 * scarcity_norm + 0.15 * dur_norm
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5.0:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
