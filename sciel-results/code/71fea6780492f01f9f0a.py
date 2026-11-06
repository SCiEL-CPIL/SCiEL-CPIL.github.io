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

    # Compute weighted rank: sum of times along longest path from node to any sink
    # This is a more nuanced measure than just counting levels
    w_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            w_rank[u] = times[u]
        else:
            w_rank[u] = times[u] + max(w_rank[s] for s in successors[u])

    # Compute transitive successor count and time sum via proper DFS
    trans_succ_count = np.zeros(n, dtype=int)
    trans_succ_time = np.zeros(n, dtype=float)

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

    # Compute "poset rank" (Cox-Johnson): weighted longest path considering all successors
    # This is the classic heuristic for ALB - sum of times on the longest path from each node
    # to the end, but weighted by the actual task times
    # For nodes with multiple successors, use the max (bottleneck path)
    poset_rank = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            poset_rank[u] = times[u]
        else:
            # Use max over successors (bottleneck path)
            poset_rank[u] = times[u] + max(poset_rank[s] for s in successors[u])

    # Also compute "mean successor rank" for a softer measure
    mean_succ_rank = np.zeros(n, dtype=float)
    for u in range(n):
        if successors[u]:
            mean_succ_rank[u] = np.mean([poset_rank[s] for s in successors[u]])
        else:
            mean_succ_rank[u] = 0.0

    # Normalize features
    max_time = times.max() if times.max() > 0 else 1.0
    max_rank = rank.max() if rank.max() > 0 else 1.0
    max_w_rank = w_rank.max() if w_rank.max() > 0 else 1.0
    max_succ = trans_succ_count.max() if trans_succ_count.max() > 0 else 1.0
    max_succ_time = trans_succ_time.max() if trans_succ_time.max() > 0 else 1.0
    max_poset = poset_rank.max() if poset_rank.max() > 0 else 1.0
    max_mean_sr = mean_succ_rank.max() if mean_succ_rank.max() > 0 else 1.0

    norm_time = times / max_time
    norm_rank = rank / max_rank
    norm_w_rank = w_rank / max_w_rank
    norm_succ = trans_succ_count.astype(float) / max_succ
    norm_succ_time = trans_succ_time / max_succ_time
    norm_poset = poset_rank / max_poset
    norm_mean_sr = mean_succ_rank / max_mean_sr

    # Priority score: combine features with emphasis on poset rank (weighted longest path)
    # The poset rank captures both depth and task duration along the critical path
    # Transitive successor count captures breadth (bottleneck-ness)
    # Mean successor rank adds a softer signal about the "weight" of downstream work
    priority_scores = (
        0.35 * norm_poset +
        0.25 * norm_succ +
        0.20 * norm_w_rank +
        0.10 * norm_mean_sr +
        0.10 * norm_time
    )

    # Add small deterministic tie-breaker
    priority_scores = priority_scores + np.arange(n) * 1e-10

    # Deadline guard
    if time.time() - t0 > DEADLINE:
        return times  # Fallback to simple time-based

    return priority_scores
# EVOLVE-BLOCK-END
