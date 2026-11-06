import numpy as np
import time
from collections import deque

def priority(instance):
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    predecessors = instance["predecessors"]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor map: succ[i] = list of tasks that depend on i
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            succ[p].append(i)
    
    # Topological sort (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If not all nodes, there's a cycle - fallback
    if len(topo_order) != n:
        return times
    
    # Compute longest path from each node to a terminal (reverse topological order)
    # Using soft-max for smoothness
    lp = [0.0] * n
    temp = 0.15
    for node in reversed(topo_order):
        if succ[node]:
            succ_lp = [lp[s] for s in succ[node]]
            m = max(succ_lp)
            exps = [np.exp((x - m) / temp) for x in succ_lp]
            soft_max_val = m + temp * np.log(sum(exps))
            lp[node] = times[node] + soft_max_val
        else:
            lp[node] = times[node]
    
    # Compute weighted precedence (total downstream work)
    wp = [0.0] * n
    for node in reversed(topo_order):
        w = times[node]
        for s in succ[node]:
            w += wp[s]
        wp[node] = w
        
    # Compute number of IMMEDIATE successors
    num_succ_immediate = [len(succ[i]) for i in range(n)]
    
    # Compute number of TRANSITIVE successors
    # trans_succ[i] = 1 for each immediate successor + sum of trans_succ of immediate successors
    trans_succ = [0] * n
    for node in reversed(topo_order):
        count = 0
        for s in succ[node]:
            count += 1 + trans_succ[s]
        trans_succ[node] = count
    
    # Compute depth: number of levels from node to deepest terminal
    depth = [0] * n
    for node in reversed(topo_order):
        if succ[node]:
            depth[node] = 1 + max(depth[s] for s in succ[node])
        else:
            depth[node] = 0
            
    # Compute sum of immediate successor times
    succ_time_sum = [0.0] * n
    for i in range(n):
        for s in succ[i]:
            succ_time_sum[i] += times[s]
            
    # Compute predecessor count
    pred_count = [len(predecessors[i]) for i in range(n)]
    
    # Normalization
    lp_min, lp_max = min(lp), max(lp)
    lp_range = lp_max - lp_min if lp_max > lp_min else 1.0
    
    wp_min, wp_max = min(wp), max(wp)
    wp_range = wp_max - wp_min if wp_max > wp_min else 1.0
    
    depth_min, depth_max = min(depth), max(depth)
    depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
    
    time_min, time_max = min(times), max(times)
    time_range = time_max - time_min if time_max > time_min else 1.0
    
    max_num_succ_imm = max(num_succ_immediate) if num_succ_immediate else 1
    max_num_succ_trans = max(trans_succ) if trans_succ else 1
    max_succ_time = max(succ_time_sum) if succ_time_sum else 1.0
    if max_succ_time == 0:
        max_succ_time = 1.0
    
    max_pred = max(pred_count) if pred_count else 1
    
    priority_scores = np.zeros(n)
    
    for i in range(n):
        lp_norm = (lp[i] - lp_min) / lp_range
        wp_norm = (wp[i] - wp_min) / wp_range
        # Use transitive successors as the primary branching metric based on insights
        trans_norm = trans_succ[i] / max_num_succ_trans
        # Keep immediate as a secondary, smaller weight
        imm_norm = num_succ_immediate[i] / max_num_succ_imm
        depth_norm = (depth[i] - depth_min) / depth_range
        time_norm = (times[i] - time_min) / time_range
        succ_time_norm = succ_time_sum[i] / max_succ_time
        pred_norm = pred_count[i] / max_pred
        
        # Adjusted weights:
        # Emphasize LP and WP strongly.
        # Shift weight from immediate successors to transitive successors.
        priority_scores[i] = (0.50 * lp_norm + 
                              0.25 * wp_norm + 
                              0.10 * time_norm +
                              0.08 * trans_norm +  # Increased weight for transitive branching
                              0.02 * imm_norm +    # Reduced weight for immediate branching
                              0.03 * depth_norm +
                              0.02 * pred_norm)
    
    # Check time budget
    if time.time() - t_start > deadline:
        return priority_scores
    
    # Tie-breaking: earlier in topological order gets slightly higher priority
    eps = 1e-12
    topo_pos = [0] * n
    for pos, node in enumerate(topo_order):
        topo_pos[node] = pos
    for i in range(n):
        priority_scores[i] += eps * (n - topo_pos[i])
        
    return priority_scores
