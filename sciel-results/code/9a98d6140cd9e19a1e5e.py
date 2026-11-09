"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the priority-based list scheduling.
    
    Strategy: Use a composite score that balances:
    1. Task duration (longer tasks first to reduce fragmentation)
    2. Number of successors (tasks with more successors should go earlier)
    3. Precedence depth (tasks deeper in the precedence graph)
    4. A 'critical path' estimate based on longest path from task to end
    
    The key insight is that we want to assign tasks that are on the critical path
    and have many successors early, while also preferring longer tasks to reduce
    the number of stations needed.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin: return within 10s
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor map
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Compute depth (longest path from start to task) and height (longest path from task to end)
    depth = [0.0] * n
    height = [0.0] * n
    
    # Topological order using Kahn's algorithm
    in_degree = pred_count[:]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    
    in_deg_copy = in_degree[:]
    q = list(queue)
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    # If topo order doesn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Fallback: just use times
        return times.copy()
    
    # Compute depth (longest path from any source to node)
    for u in topo_order:
        for v in successors[u]:
            if depth[v] < depth[u] + times[u]:
                depth[v] = depth[u] + times[u]
    
    # Compute height (longest path from node to any sink) - reverse topological
    for u in reversed(topo_order):
        for v in successors[u]:
            if height[u] < times[u] + height[v]:
                height[u] = times[u] + height[v]
    
    # Number of successors (immediate and transitive)
    succ_count = [0] * n
    # Use DFS with memoization for transitive successor count
    memo = [-1] * n
    
    def count_succs(u):
        if memo[u] >= 0:
            return memo[u]
        total = 0
        for v in successors[u]:
            total += 1 + count_succs(v)
        memo[u] = total
        return total
    
    for i in range(n):
        succ_count[i] = count_succs(i)
    
    # Normalize components
    max_time = np.max(times) if n > 0 else 1.0
    max_depth = np.max(depth) if n > 0 else 1.0
    max_height = np.max(height) if n > 0 else 1.0
    max_succ = np.max(succ_count) if n > 0 else 1.0
    
    if max_time == 0: max_time = 1.0
    if max_depth == 0: max_depth = 1.0
    if max_height == 0: max_height = 1.0
    if max_succ == 0: max_succ = 1.0
    
    # Composite score: higher is better priority
    # Weight: height (critical path remaining) + time + successor count
    score = (
        1.0 * (height / max_height) +
        0.5 * (times / max_time) +
        0.3 * (succ_count / max_succ)
    )
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
