"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_floats(instance):
    """Compute total float for each activity using CPM."""
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors and successors
    if "predecessors" in instance:
        preds = instance["predecessors"]
    elif "predecessor" in instance:
        preds = instance["predecessor"]
    else:
        preds = [[] for _ in range(n)]
    
    # Build adjacency
    succs = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
            in_degree[i] += 1
    
    # Topological sort
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_indeg = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            temp_indeg[s] -= 1
            if temp_indeg[s] == 0:
                queue.append(s)
    
    # Forward pass (earliest start)
    es = [0.0] * n
    for node in topo_order:
        for s in succs[node]:
            es[s] = max(es[s], es[node] + durations[node])
    
    # Project duration
    project_end = max(es[i] + durations[i] for i in range(n)) if n > 0 else 0.0
    
    # Backward pass (latest start)
    ls = [0.0] * n
    for node in reversed(topo_order):
        # Latest finish is min of successors' latest start
        lf = project_end
        for s in succs[node]:
            lf = min(lf, ls[s])
        ls[node] = lf - durations[node]
    
    # Total float = LS - ES
    tf = [ls[i] - es[i] for i in range(n)]
    return tf

def _compute_successor_depth(instance):
    """Compute the longest path from each activity to any terminal activity."""
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if "predecessors" in instance:
        preds = instance["predecessors"]
    elif "predecessor" in instance:
        preds = instance["predecessor"]
    else:
        preds = [[] for _ in range(n)]
    
    # Build successors
    succs = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute longest path from each node to any terminal (successor depth)
    # Use memoized DFS
    depth = [0.0] * n
    visited = [False] * n
    
    def compute_depth(node):
        if visited[node]:
            return depth[node]
        visited[node] = True
        if not succs[node]:
            depth[node] = durations[node]
        else:
            max_succ_depth = 0.0
            for s in succs[node]:
                max_succ_depth = max(max_succ_depth, compute_depth(s))
            depth[node] = durations[node] + max_succ_depth
        return depth[node]
    
    for i in range(n):
        if not visited[i]:
            compute_depth(i)
    
    return depth

def priority(instance):
    """
    Priority score: higher = scheduled earlier.
    Uses combination of:
    1. Longest duration (LDT)
    2. Minimum total float (MTF) - critical path activities first
    3. High resource demand pressure
    4. Successor depth (longest path to terminal)
    """
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin before 150s kill
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
    
    # Get demands and capacities
    if "demands" in instance:
        demands = instance["demands"]  # shape: (n, num_resources) or list of lists
    else:
        demands = None
    
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = None
    
    # Compute total float
    try:
        tf = _compute_floats(instance)
        tf_arr = np.array(tf)
        # Normalize float: lower float = higher priority
        if tf_arr.max() > tf_arr.min():
            float_priority = (tf_arr.max() - tf_arr) / (tf_arr.max() - tf_arr.min())
        else:
            float_priority = np.ones(n)
    except Exception:
        float_priority = np.ones(n)
    
    # Duration priority: longer = higher
    if durations.max() > durations.min():
        dur_priority = (durations - durations.min()) / (durations.max() - durations.min())
    else:
        dur_priority = np.ones(n)
    
    # Successor depth: activities with longer downstream paths get higher priority
    try:
        succ_depth = _compute_successor_depth(instance)
        succ_depth_arr = np.array(succ_depth)
        if succ_depth_arr.max() > succ_depth_arr.min():
            depth_priority = (succ_depth_arr - succ_depth_arr.min()) / (succ_depth_arr.max() - succ_depth_arr.min())
        else:
            depth_priority = np.ones(n)
    except Exception:
        depth_priority = np.ones(n)
    
    # Resource pressure: if an activity uses a lot of a scarce resource, prioritize it
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        try:
            demands_arr = np.asarray(demands, dtype=float)
            if demands_arr.ndim == 1:
                demands_arr = demands_arr.reshape(-1, 1)
            num_resources = demands_arr.shape[1] if demands_arr.ndim == 2 else 1
            
            if num_resources > 0 and len(capacities) > 0:
                # For each activity, compute max ratio of demand to capacity across resources
                # High ratio means the activity is resource-intensive
                for i in range(n):
                    if demands_arr[i].max() > 0:
                        ratios = demands_arr[i] / capacities
                        resource_pressure[i] = ratios.max()
                
                if resource_pressure.max() > resource_pressure.min():
                    resource_pressure = (resource_pressure - resource_pressure.min()) / (resource_pressure.max() - resource_pressure.min())
        except Exception:
            resource_pressure = np.zeros(n)
    
    # Combine priorities with weights
    # Weight: float (critical path) most important, then depth, then duration, then resource pressure
    w_float = 0.4
    w_depth = 0.25
    w_dur = 0.2
    w_res = 0.15
    
    scores = w_float * float_priority + w_depth * depth_priority + w_dur * dur_priority + w_res * resource_pressure
    
    return scores
# EVOLVE-BLOCK-END
