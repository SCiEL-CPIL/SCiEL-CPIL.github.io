import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance.get("cycle_time", 40.0))
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)
    
    # Compute topological order using Kahn's algorithm
    in_degree = [len(predecessors[i]) for i in range(n)]
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
    
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    
    # If topological sort didn't include all nodes (cycle), fall back to simple order
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Compute Critical Path (longest path from node to sink)
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            if times[u] + cp[v] > cp[u]:
                cp[u] = times[u] + cp[v]
    
    # Compute upstream path (longest path from source to node)
    up = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in successors[u]:
            if up[u] + times[v] > up[v]:
                up[v] = up[u] + times[v]
    
    # Count transitive successors for each task
    # This measures how much "work" each task unlocks
    transitive_succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        if time.time() > deadline:
            break
        # BFS/DFS to count all transitive successors
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for s in successors[node]:
                if s not in visited:
                    visited.add(s)
                    stack.append(s)
        transitive_succ_count[i] = len(visited)
    
    # Count immediate successors
    num_immediate_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Count immediate predecessors
    num_immediate_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)
    
    # Bottleneck slack heuristic:
    # Favor tasks with small (cycle_time - time_i) / branching factor
    # This prioritizes tasks that consume most capacity relative to what they unlock
    denom = np.maximum(num_immediate_succ, 1.0)
    slack = (cycle_time - times) / denom
    bottleneck_score = -slack  # Invert so higher priority = lower slack
    
    # Combine heuristics:
    # 1. Critical Path (downstream) - primary driver
    # 2. Upstream path - secondary
    # 3. Transitive successor count - rewards tasks that unlock lots of work
    # 4. Bottleneck slack - rewards capacity-consuming tasks
    # 5. Predecessor count - slight penalty for tasks with many preds (later in DAG)
    # 6. Task time - tiny tie-breaker
    
    # Normalize some components to prevent scale issues
    cp_max = np.max(cp) if np.max(cp) > 0 else 1.0
    up_max = np.max(up) if np.max(up) > 0 else 1.0
    ts_max = np.max(transitive_succ_count) if np.max(transitive_succ_count) > 0 else 1.0
    
    priorities = (
        1.0 * (cp / cp_max) +
        0.4 * (up / up_max) +
        0.3 * (transitive_succ_count / ts_max) +
        0.2 * (bottleneck_score / (np.max(np.abs(bottleneck_score)) + 1e-9)) +
        0.1 * (num_immediate_pred / (np.max(num_immediate_pred) + 1e-9)) +
        times * 1e-6
    )
    
    return priorities
