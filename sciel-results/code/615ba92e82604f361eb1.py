"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    """
    Score each assembly task for the priority rule used by the station-fill heuristic.
    Strategy: 'positional criticality' + downstream pressure.
    - Compute earliest start time (EST) for each task via longest path from sources.
    - Compute number of transitive successors (downstream pressure).
    - Priority = w1 * (EST / cycle_time) + w2 * (normalized successor count) + w3 * (task_time / cycle_time)
    Higher priority tasks are placed first.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", {})
    cycle_time = float(instance.get("cycle_time", 40.0))
    n = len(times)

    # Build adjacency: pred -> list of successors
    succ = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    for task in range(n):
        plist = preds.get(task, [])
        for p in plist:
            succ[p].append(task)
            in_deg[task] += 1

    # Topological order (Kahn's algorithm)
    in_deg_work = in_deg.copy()
    q = deque([i for i in range(n) if in_deg_work[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            in_deg_work[v] -= 1
            if in_deg_work[v] == 0:
                q.append(v)

    # If topo order doesn't cover all nodes (cycle), fall back to simple sort
    if len(topo) != n:
        topo = list(range(n))

    # Compute EST (earliest start time) = longest path from any source to task
    est = np.zeros(n, dtype=float)
    for u in topo:
        for v in succ[u]:
            if est[u] + times[u] > est[v]:
                est[v] = est[u] + times[u]

    # Compute transitive successor counts via reverse topological order
    # succ_count[i] = number of nodes reachable from i (excluding i itself)
    succ_count = np.zeros(n, dtype=int)
    for u in reversed(topo):
        # succ_count[u] = sum over direct successors v of (1 + succ_count[v])
        # But this overcounts if multiple paths lead to same node.
        # For a DAG, we need to count distinct reachable nodes.
        # For simplicity and speed, use a BFS/DFS per node if n is small,
        # or approximate by sum of (1 + succ_count[v]) for direct successors (overcounts shared successors).
        # Since n=30, we can do exact BFS for each node.
        pass

    # Exact transitive successor count via BFS for each node (n=30, so 30 BFS is fine)
    succ_count = np.zeros(n, dtype=int)
    for u in range(n):
        visited = set()
        dq = deque(succ[u])
        while dq:
            v = dq.popleft()
            if v not in visited:
                visited.add(v)
                dq.extend(succ[v])
        succ_count[u] = len(visited)

    # Normalize
    max_est = np.max(est) if n > 0 and np.max(est) > 0 else 1.0
    max_succ = np.max(succ_count) if n > 0 and np.max(succ_count) > 0 else 1.0
    max_time = np.max(times) if n > 0 and np.max(times) > 0 else 1.0

    est_norm = est / max_est
    succ_norm = succ_count.astype(float) / max_succ
    time_norm = times / max_time

    # Weighted combination
    w1 = 0.4  # positional criticality
    w2 = 0.4  # downstream pressure
    w3 = 0.2  # task duration

    scores = w1 * est_norm + w2 * succ_norm + w3 * time_norm

    return scores
# EVOLVE-BLOCK-END
