"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task. Stations are filled with the highest-priority
    eligible task that fits the cycle time. We want to minimize the number of
    stations.

    Strategy:
    - Compute a "rank" based on longest path to end (critical path length from
      the task to the end of the line).
    - Also consider task duration.
    - Combine: higher priority if it has larger downstream critical path, and
      among those, prefer longer tasks (to fill stations tightly).
    - Use a small epsilon-based ordering to break ties deterministically.

    The key heuristic for assembly line balancing:
    1. Prioritize tasks that have many successors (they are "critical" to the
       flow).
    2. Among critical tasks, prefer longer durations to pack stations.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    n = len(times)
    cycle_time = instance["cycle_time"]

    # Build successor lists
    succs = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in preds[j]:
            succs[i].append(j)
            in_degree[j] += 1

    # Longest path from each node to the end (reverse topological)
    # Use topological order
    from collections import deque
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg = in_degree.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # Longest path from node to end
    lpt = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if succs[u]:
            lpt[u] = times[u] + max(lpt[v] for v in succs[u])
        else:
            lpt[u] = times[u]

    # Also compute longest path from start to node (for reference)
    lps = np.zeros(n, dtype=float)
    for u in topo:
        if preds[u]:
            lps[u] = times[u] + max(lps[p] for p in preds[u])
        else:
            lps[u] = times[u]

    # Number of successors (direct + transitive)
    # Compute transitive successor count via reverse topo
    succ_count = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if succs[u]:
            # Count unique successors transitively
            seen = set()
            stack = list(succs[u])
            while stack:
                v = stack.pop()
                if v not in seen:
                    seen.add(v)
                    stack.extend(succs[v])
            succ_count[u] = len(seen)
        else:
            succ_count[u] = 0

    # Priority: primary = lpt (longest path to end), secondary = time,
    # tertiary = number of successors
    # Scale to [0,1] ranges
    lpt_min, lpt_max = lpt.min(), lpt.max()
    t_min, t_max = times.min(), times.max()
    sc_min, sc_max = succ_count.min(), succ_count.max()

    if lpt_max > lpt_min:
        lpt_norm = (lpt - lpt_min) / (lpt_max - lpt_min)
    else:
        lpt_norm = np.zeros(n)

    if t_max > t_min:
        t_norm = (times - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros(n)

    if sc_max > sc_min:
        sc_norm = (succ_count - sc_min) / (sc_max - sc_min)
    else:
        sc_norm = np.zeros(n)

    # Weighted combination: emphasize critical path, then duration, then fanout
    priority_scores = 1.0 * lpt_norm + 0.3 * t_norm + 0.1 * sc_norm

    return priority_scores
# EVOLVE-BLOCK-END
