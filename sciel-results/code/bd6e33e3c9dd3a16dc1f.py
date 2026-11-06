"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    # Deadline guard: we must finish well within 120s.
    # For 30 tasks, even O(n^2) or O(n^3) is trivial, so no heavy search needed.
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance.get("predecessors", [])
    cycle_time = instance.get("cycle_time", 40.0)
    n = len(times)
    
    # Build adjacency: pred[i] is list of predecessors of task i
    # We need transitive successors and predecessors.
    
    # For each task, compute:
    # 1. sum of times of all transitive predecessors (inclusive)
    # 2. sum of times of all transitive successors (inclusive)
    # 3. count of transitive successors
    # 4. count of transitive predecessors
    
    # Build successor list
    successors = [[] for _ in range(n)]
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            pred_list[i].append(p)
    
    # Compute transitive closure using DFS/BFS
    # trans_succ[i] = set of all transitive successors of i (not including i)
    # trans_pred[i] = set of all transitive predecessors of i (not including i)
    
    trans_succ = [set() for _ in range(n)]
    trans_pred = [set() for _ in range(n)]
    
    # Compute transitive successors via reverse topological order
    # First, get topological order
    in_degree = [len(pred_list[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes are in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Compute transitive successors in reverse topological order
    for u in reversed(topo_order):
        for v in successors[u]:
            trans_succ[u].add(v)
            trans_succ[u].update(trans_succ[v])
    
    # Compute transitive predecessors
    for u in topo_order:
        for p in pred_list[u]:
            trans_pred[u].add(p)
            trans_pred[u].update(trans_pred[p])
    
    # Now compute priority scores
    # Strategy: "resource contention" metric
    # priority[i] = sum of times of transitive predecessors + sum of times of transitive successors
    # Plus a bonus for number of transitive successors (bottleneck tasks)
    
    pred_time_sum = np.zeros(n)
    succ_time_sum = np.zeros(n)
    succ_count = np.zeros(n)
    pred_count = np.zeros(n)
    
    for i in range(n):
        pred_time_sum[i] = sum(times[p] for p in trans_pred[i])
        succ_time_sum[i] = sum(times[s] for s in trans_succ[i])
        succ_count[i] = len(trans_succ[i])
        pred_count[i] = len(trans_pred[i])
    
    # Normalize by cycle time
    ct = max(cycle_time, 1.0)
    
    # The idea: tasks that are part of dense, interconnected subgraphs should be prioritized
    # because placing them early allows more flexibility in packing successors.
    # 
    # Combine:
    # - succ_time_sum / ct: temporal weight of downstream work
    # - pred_time_sum / ct: temporal weight of upstream work  
    # - succ_count: number of downstream tasks (bottleneck indicator)
    # - times[i]: the task's own duration
    #
    # Weighted combination:
    # We want to prioritize tasks that have many successors (bottleneck) AND are in dense regions.
    
    # Scale factors to balance terms
    # succ_time_sum ranges from 0 to sum(times), normalize by max
    max_succ_time = np.max(succ_time_sum) if np.max(succ_time_sum) > 0 else 1.0
    max_pred_time = np.max(pred_time_sum) if np.max(pred_time_sum) > 0 else 1.0
    max_succ_count = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    max_pred_count = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    
    # Priority: emphasize successors (bottleneck) and global connectivity
    # Higher priority = placed earlier
    # 
    # Use a combination:
    # p = w1 * (succ_time_sum / max_succ_time) + w2 * (pred_time_sum / max_pred_time) 
    #     + w3 * (succ_count / max_succ_count) + w4 * (pred_count / max_pred_count)
    #     + w5 * (times / max(times))
    
    # The key insight: tasks with many successors should go early so their successors become eligible sooner
    # Tasks with large downstream time should go early to reduce remaining work
    
    w_succ_time = 2.0
    w_pred_time = 1.0
    w_succ_count = 1.5
    w_pred_count = 0.5
    w_time = 0.5
    
    max_t = np.max(times) if np.max(times) > 0 else 1.0
    
    scores = (w_succ_time * (succ_time_sum / max_succ_time)
              + w_pred_time * (pred_time_sum / max_pred_time)
              + w_succ_count * (succ_count / max_succ_count)
              + w_pred_count * (pred_count / max_pred_count)
              + w_time * (times / max_t))
    
    return scores
# EVOLVE-BLOCK-END
