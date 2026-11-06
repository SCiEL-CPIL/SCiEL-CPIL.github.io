"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s; keep a safety margin.
_DEADLINE = 140.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def priority(instance):
    """
    Score each assembly task for a priority-based line balancing heuristic.
    Stations are filled with the highest-priority eligible task that fits the cycle time.
    We want to minimize the mean number of stations over 5 generated 30-task instances.

    Strategy: Use a composite priority that combines:
      1. Longest task time (larger tasks first to fill stations efficiently)
      2. Number of successors (tasks with more successors should go earlier)
      3. Position in the precedence graph (topological depth from start)

    This is a static priority rule evaluated once per instance.
    """
    if _elapsed() > _DEADLINE - 5.0:
        # Safety: return a simple fallback quickly
        return np.asarray(instance["times"], dtype=float)

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance.get("predecessors", [])
    cycle_time = instance.get("cycle_time", 40.0)

    # Build adjacency: pred -> list of successors
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            succ[p].append(i)
            indeg[i] += 1

    # Compute longest path from each node to any sink (successor depth)
    # Using topological order (Kahn's algorithm)
    from collections import deque
    q = deque()
    for i in range(n):
        if indeg[i] == 0:
            q.append(i)

    top_order = []
    indeg_tmp = indeg.copy()
    while q:
        u = q.popleft()
        top_order.append(u)
        for v in succ[u]:
            indeg_tmp[v] -= 1
            if indeg_tmp[v] == 0:
                q.append(v)

    # If topological sort didn't cover all nodes (cycle), fall back
    if len(top_order) < n:
        # Fallback: just use times
        return times

    # longest_successor_time[i] = max over paths starting at i of sum of times
    # Compute in reverse topological order
    longest_succ = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        if succ[u]:
            longest_succ[u] = times[u] + np.max([longest_succ[v] for v in succ[u]])
        else:
            longest_succ[u] = times[u]

    # Number of successors (direct + transitive)
    num_succ = np.zeros(n, dtype=int)
    for u in reversed(top_order):
        if succ[u]:
            num_succ[u] = 1 + np.max([num_succ[v] for v in succ[u]]) if len(succ[u]) > 1 else num_succ[succ[u][0]] + 1
        # Actually let's do it properly: count all transitive successors
    # Recompute num_succ properly using DP
    num_succ = np.zeros(n, dtype=int)
    for u in reversed(top_order):
        if succ[u]:
            # Count unique successors via DP isn't straightforward for transitive count
            # Use a simpler metric: direct successors count + longest path length
            num_succ[u] = len(succ[u])
        # For a better metric, use the number of nodes in the subgraph reachable from u
    # Let's use a different approach: position in topological order (earlier = higher priority)
    # Normalize topological position
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        topo_pos[u] = idx

    # Composite priority:
    # - Higher times should be prioritized
    # - Tasks with longer successor paths should go earlier
    # - Earlier in topological order should be slightly preferred
    #
    # Normalize each component to [0, 1]
    t_min, t_max = times.min(), times.max()
    t_range = t_max - t_min if t_max > t_min else 1.0
    norm_time = (times - t_min) / t_range

    ls_min, ls_max = longest_succ.min(), longest_succ.max()
    ls_range = ls_max - ls_min if ls_max > ls_min else 1.0
    norm_ls = (longest_succ - ls_min) / ls_range

    # Weighted combination: emphasize time and successor path length
    # The key insight: placing large tasks early fills stations with big chunks,
    # reducing fragmentation. Also, tasks that are bottlenecks (long successor chains)
    # should be placed early so their successors can follow.
    score = 0.5 * norm_time + 0.35 * norm_ls + 0.15 * (topo_pos / max(n - 1, 1))

    # Add a small bonus for tasks that are "critical" in terms of cycle time utilization
    # Tasks close to cycle_time are hard to pack, so prioritize them
    ct_util = times / max(cycle_time, 1e-9)
    ct_util_norm = ct_util / max(ct_util.max(), 1e-9)
    score += 0.1 * ct_util_norm

    return score
# EVOLVE-BLOCK-END
