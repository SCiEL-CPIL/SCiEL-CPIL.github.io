"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance.get("cycle_time", 40.0)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)

    # Compute in-degrees
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])

    # Topological order (Kahn's algorithm)
    topo_order = []
    q = deque()
    in_deg_copy = in_degree.copy()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    while q:
        u = q.popleft()
        topo_order.append(u)
        for s in successors[u]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                q.append(s)

    # If topo_order doesn't cover all nodes, append remaining (shouldn't happen for valid DAG)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))

    # Compute rank (longest path from node to any sink)
    # Process in reverse topological order
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            rank[u] = 0.0
        else:
            rank[u] = 1.0 + max(rank[s] for s in successors[u])

    # Compute transitive successor count and time sum
    # For each node, count all transitive successors and sum their times
    trans_succ_count = np.zeros(n, dtype=int)
    trans_succ_time = np.zeros(n, dtype=float)

    # Process in reverse topological order so that successors are already computed
    for u in reversed(topo_order):
        for s in successors[u]:
            # Add direct successor's contributions
            trans_succ_count[u] += 1 + trans_succ_count[s]
            trans_succ_time[u] += times[s] + trans_succ_time[s]
            # But we need to avoid double-counting if multiple paths lead to same node
            # This is a DAG, so we need a proper transitive closure approach
            pass

    # Redo with proper transitive closure (for n=30, this is fine)
    # For each node, find all transitive successors via DFS
    for i in range(n):
        visited = set()
        stack = list(successors[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                trans_succ_count[i] += 1
                trans_succ_time[i] += times[node]
                for s in successors[node]:
                    if s not in visited:
                        stack.append(s)

    # Compute "bottleneck score" - how critical each task is
    # A task is more critical if it has many successors and those successors have long total time
    # This captures the idea that bottleneck tasks should be placed early

    # Also compute the "remaining work" after each task
    # remaining_work[i] = sum of times of all transitive successors of i
    # This is already trans_succ_time[i]

    # Normalize features
    max_time = times.max() if times.max() > 0 else 1.0
    max_rank = rank.max() if rank.max() > 0 else 1.0
    max_succ = trans_succ_count.max() if trans_succ_count.max() > 0 else 1.0
    max_succ_time = trans_succ_time.max() if trans_succ_time.max() > 0 else 1.0

    norm_time = times / max_time
    norm_rank = rank / max_rank
    norm_succ = trans_succ_count.astype(float) / max_succ
    norm_succ_time = trans_succ_time / max_succ_time

    # Priority score: combine features
    # Key insight: prioritize tasks with many transitive successors (bottlenecks)
    # and tasks that are deep in the precedence chain (early tasks)
    priority_scores = (
        0.35 * norm_rank +
        0.30 * norm_succ +
        0.20 * norm_time +
        0.15 * norm_succ_time
    )

    # Add small deterministic tie-breaker
    priority_scores = priority_scores + np.arange(n) * 1e-10

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
