"""Rewritten program for Resource-Constrained Project Scheduling (serial SGS).

Approach:
1. Compute static priority using a blend of:
   - Longest path to end (critical path / total float inverse)
   - Total resource demand (scarce resource pressure)
   - Number of successors (MPS - most preceding successors)
2. Use this priority in the serial schedule generation scheme (SGS).
3. The priority function returns one score per activity; higher score = higher priority.

Key insight: For RCPSP, scheduling activities with longer remaining paths to the project
end first (i.e., low total float) tends to reduce makespan. Combining with resource
pressure and successor count helps break ties and handle resource conflicts better.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute priority scores for each activity.
    Higher score = should be scheduled earlier in serial SGS.
    
    instance contains:
      - durations: list/array of activity durations
      - demands: list of arrays, one per resource, giving demand per activity
      - capacities: list/array of resource capacities
      - predecessors: list of lists, predecessors[i] = list of activity indices that must finish before i starts
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    preds_raw = instance.get("predecessors", None)
    if preds_raw is None:
        preds_raw = [[] for _ in range(n)]
    
    # Normalize predecessors to lists of ints
    preds = []
    for i in range(n):
        p = preds_raw[i]
        if p is None:
            p = []
        preds.append([int(x) for x in p])
    
    # Build successor lists
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
    
    # Compute earliest start times (forward pass) - for CPM
    est = np.zeros(n, dtype=float)
    # Topological order using Kahn's algorithm
    in_degree = [len(preds[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Forward pass for EST
    for u in topo_order:
        for v in succs[u]:
            est[v] = max(est[v], est[u] + durations[u])
    
    # Latest start times (backward pass)
    lft = np.zeros(n, dtype=float)
    # Project end time
    makespan_cpm = np.max(est + durations) if n > 0 else 0.0
    lft[:] = makespan_cpm
    
    # Reverse topological order
    for u in reversed(topo_order):
        for v in succs[u]:
            lft[u] = min(lft[u], lft[v] - durations[u])
    
    # Total float (slack) = LST - EST
    total_float = lft - est
    
    # Component 1: Inverse total float (activities with less slack are more critical)
    # Normalize to [0, 1]
    if np.max(total_float) > np.min(total_float):
        tf_norm = (np.max(total_float) - total_float) / (np.max(total_float) - np.min(total_float))
    else:
        tf_norm = np.ones(n, dtype=float)
    
    # Component 2: Total resource demand (weighted by scarcity)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        # demands is a list of arrays (one per resource), or a 2D array
        if isinstance(demands, (list, tuple)):
            demand_arr = np.array(demands, dtype=float)
        else:
            demand_arr = np.asarray(demands, dtype=float)
        
        if demand_arr.ndim == 1:
            demand_arr = demand_arr.reshape(-1, 1)
        
        cap_arr = np.asarray(capacities, dtype=float)
        if cap_arr.ndim == 0:
            cap_arr = cap_arr.reshape(1)
        
        # Scarcity weight for each resource: 1/capacity (higher when capacity is low)
        scarcity = 1.0 / np.maximum(cap_arr, 1e-9)
        
        # Total weighted demand per activity
        # demand_arr shape: (n_activities, n_resources) or (n_resources, n_activities)
        if demand_arr.shape[0] == n:
            # (n_activities, n_resources)
            weighted_demand = demand_arr @ scarcity
        else:
            # (n_resources, n_activities)
            weighted_demand = scarcity @ demand_arr
        
        # Normalize
        if np.max(weighted_demand) > np.min(weighted_demand):
            wd_norm = (weighted_demand - np.min(weighted_demand)) / (np.max(weighted_demand) - np.min(weighted_demand))
        else:
            wd_norm = np.zeros(n, dtype=float)
    else:
        wd_norm = np.zeros(n, dtype=float)
    
    # Component 3: Most Preceding Successors (MPS) - number of direct + indirect successors
    # Count all successors (transitive)
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        # BFS/DFS to count all successors
        visited = set()
        stack = [i]
        while stack:
            u = stack.pop()
            for v in succs[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        succ_count[i] = len(visited)
    
    if np.max(succ_count) > np.min(succ_count):
        sc_norm = (succ_count - np.min(succ_count)) / (np.max(succ_count) - np.min(succ_count))
    else:
        sc_norm = np.zeros(n, dtype=float)
    
    # Component 4: Duration (longer activities first)
    if np.max(durations) > np.min(durations):
        dur_norm = (durations - np.min(durations)) / (np.max(durations) - np.min(durations))
    else:
        dur_norm = np.zeros(n, dtype=float)
    
    # Combine components with weights
    # Critical path (inverse float) is most important, then resource pressure, then MPS, then duration
    w_tf = 0.45
    w_wd = 0.25
    w_sc = 0.15
    w_dur = 0.15
    
    scores = w_tf * tf_norm + w_wd * wd_norm + w_sc * sc_norm + w_dur * dur_norm
    
    return scores
# EVOLVE-BLOCK-END
