"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _compute_transitive_successors(predecessors, n):
    """
    Compute the set of all transitive successors for each task.
    Uses BFS/DFS from each node on the reversed graph.
    """
    # Build successor adjacency
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    trans_succ = [set() for _ in range(n)]
    
    # For each node, find all reachable nodes
    for i in range(n):
        visited = set()
        queue = deque(successors[i])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for s in successors[node]:
                    if s not in visited:
                        queue.append(s)
        trans_succ[i] = visited
        
    return trans_succ

def _compute_ranking(instance):
    """
    Compute a sophisticated priority score for each task.
    Combines:
    1. Longest processing time (LPT) - base
    2. Number of transitive successors (unlocks more work)
    3. Critical path length (downstream workload)
    4. Bottleneck slack: (cycle_time - time) / (1 + trans_succ_count)
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor map
    successors = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute transitive successors
    trans_succ = _compute_transitive_successors(predecessors, n)
    trans_succ_counts = np.array([len(s) for s in trans_succ], dtype=float)
    
    # Compute critical path (longest path from each node to any sink)
    # Use topological order
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    topo_order = []
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    temp_in_degree = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    # Compute longest path from each node to a sink (critical path downstream)
    # Process in reverse topological order
    crit_path = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ = 0
        for s in successors[node]:
            max_succ = max(max_succ, crit_path[s])
        crit_path[node] = times[node] + max_succ
    
    # Compute slack: how much "room" is left in cycle time relative to downstream complexity
    # Small slack = high priority (tight fit, unlocks many tasks)
    epsilon = 1e-6
    slack = (cycle_time - times + epsilon) / (trans_succ_counts + 1.0)
    
    # Combine heuristics:
    # Primary: processing time (longer tasks first)
    # Secondary: number of transitive successors (unlock more work)
    # Tertiary: critical path length (more downstream work)
    # Penalty: large slack (wastes capacity)
    
    # Normalize components to [0, 1] range for better weighting
    max_time = np.max(times) if np.max(times) > 0 else 1
    max_trans = np.max(trans_succ_counts) if np.max(trans_succ_counts) > 0 else 1
    max_crit = np.max(crit_path) if np.max(crit_path) > 0 else 1
    max_slack = np.max(slack) if np.max(slack) > 0 else 1
    
    norm_time = times / max_time
    norm_trans = trans_succ_counts / max_trans
    norm_crit = crit_path / max_crit
    norm_slack = slack / max_slack
    
    # Weighted combination
    # Emphasize: time (0.4), transitive successors (0.3), critical path (0.2), inverse slack (0.1)
    scores = (0.4 * norm_time + 
              0.3 * norm_trans + 
              0.2 * norm_crit + 
              0.1 * (1.0 - norm_slack)) * max_time
    
    # Ensure positive scores
    scores = np.maximum(scores, 0.1)
    
    return scores

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    try:
        scores = _compute_ranking(instance)
    except Exception:
        # Fallback to simple LPT
        times = np.asarray(instance["times"], dtype=float)
        scores = times
        
    if time.time() > deadline:
        times = np.asarray(instance["times"], dtype=float)
        return times
        
    return scores
# EVOLVE-BLOCK-END
