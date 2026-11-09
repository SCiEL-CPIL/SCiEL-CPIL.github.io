"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _topological_sort(n, preds):
    """Return a topological ordering of tasks 0..n-1 given predecessors list."""
    # preds[i] is a list/set of tasks that must precede i
    in_degree = [0] * n
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1
            children[p].append(i)
    
    # Use Kahn's algorithm
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    order = []
    while queue:
        # To be deterministic, we could sort, but for speed just pop
        u = queue.popleft()
        order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes visited, there's a cycle; append remaining
    if len(order) < n:
        remaining = [i for i in range(n) if i not in set(order)]
        order.extend(remaining)
    
    return order

def _compute_priority(instance):
    """
    Compute priority scores for assembly line balancing.
    Strategy: Use a combination of task duration and position in the dependency graph.
    We want to prioritize tasks that are "critical" or have many successors (successor count).
    Common heuristics:
    1. Longest Processing Time (LPT) - simple, already baseline.
    2. Most Successors (MS) - tasks with many successors should be done early.
    3. Rank Position (RP) - topological rank.
    4. Critical Path - longest path from task to end.
    
    Let's try a combination: Critical Path Length (backward pass) + Successor Count.
    The critical path length (longest path from i to any sink) is a strong indicator of urgency.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build adjacency list for successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    
    # Compute longest path from each node to any sink (backward pass)
    # Use topological order (reverse) for DP
    topo_order = _topological_sort(n, preds)
    rev_topo = topo_order[::-1]
    
    longest_path = np.zeros(n)
    for u in rev_topo:
        if successors[u]:
            longest_path[u] = times[u] + max(longest_path[v] for v in successors[u])
        else:
            longest_path[u] = times[u]
    
    # Count successors (transitive? or direct?)
    # Direct successors is simpler. Let's compute transitive successor count for a stronger signal.
    # Actually, just direct might be enough. Let's try direct successor count first.
    succ_count = np.array([len(successors[i]) for i in range(n)])
    
    # Combine: 
    # Priority = alpha * normalized_longest_path + beta * normalized_succ_count
    # Higher is better priority.
    
    # Normalize to [0, 1]
    def normalize(arr):
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    lp_norm = normalize(longest_path)
    sc_norm = normalize(succ_count)
    
    # Weights: Critical path is usually more important
    alpha = 0.7
    beta = 0.3
    
    priority_scores = alpha * lp_norm + beta * sc_norm
    
    # Add a small tie-breaker based on time to ensure stability
    # But we want higher time to be higher priority generally, which is already in LP.
    
    return priority_scores

def priority(instance):
    """
    Compute priority for each task.
    Returns an array of scores, one per task.
    """
    start_time = time.time()
    
    try:
        scores = _compute_priority(instance)
    except Exception as e:
        # Fallback to simple LPT if anything goes wrong
        try:
            scores = np.asarray(instance["times"], dtype=float)
        except:
            scores = np.zeros(len(instance["times"]))
    
    return scores
# EVOLVE-BLOCK-END
