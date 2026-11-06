"""Assembly line balancing with bottleneck slack priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute number of immediate successors
    num_successors = np.array([len(succs) for succs in successors], dtype=float)
    
    # Compute downstream critical path length (longest path from task to any terminal)
    # Use topological order (reverse)
    # First, compute in-degrees for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
    
    # If not all nodes in topo order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use index order
        topo_order = list(range(n))
    
    # Compute downstream path length in reverse topological order
    down_path = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ_path = 0.0
        for succ in successors[node]:
            max_succ_path = max(max_succ_path, down_path[succ])
        down_path[node] = times[node] + max_succ_path
    
    # Compute upstream depth (number of predecessors in transitive closure approximation)
    # Use topological order to compute depth
    up_depth = np.zeros(n, dtype=float)
    for node in topo_order:
        max_pred_depth = 0.0
        for pred in predecessors[node]:
            max_pred_depth = max(max_pred_depth, up_depth[pred])
        up_depth[node] = max_pred_depth + 1
    
    # Slack: cycle_time - time_i
    slack = cycle_time - times
    slack = np.maximum(slack, 1e-9)  # Avoid division by zero or negative
    
    # Feature 1: Inverse bottleneck ratio (original heuristic)
    epsilon = 1e-9
    bottleneck_ratio = slack / (1.0 + num_successors)
    feat_bottleneck = 1.0 / (bottleneck_ratio + epsilon)
    
    # Feature 2: Downstream critical path (higher = more important to schedule early)
    max_down = np.max(down_path) if n > 0 else 1.0
    feat_downstream = down_path / (max_down + epsilon)
    
    # Feature 3: Task time (longer tasks should be prioritized)
    max_time = np.max(times) if n > 0 else 1.0
    feat_time = times / (max_time + epsilon)
    
    # Feature 4: Upstream depth (tasks deeper in the network may need careful placement)
    max_up = np.max(up_depth) if n > 0 else 1.0
    feat_upstream = up_depth / (max_up + epsilon)
    
    # Normalize bottleneck feature
    max_bottleneck = np.max(feat_bottleneck) if n > 0 else 1.0
    feat_bottleneck_norm = feat_bottleneck / (max_bottleneck + epsilon)
    
    # Combine features with weights
    # Weight downstream path heavily (critical path matters most)
    # Then time, then bottleneck slack, then upstream depth
    w_downstream = 0.35
    w_time = 0.25
    w_bottleneck = 0.30
    w_upstream = 0.10
    
    priority_scores = (w_downstream * feat_downstream + 
                       w_time * feat_time + 
                       w_bottleneck * feat_bottleneck_norm + 
                       w_upstream * feat_upstream)
    
    return priority_scores
# EVOLVE-BLOCK-END
