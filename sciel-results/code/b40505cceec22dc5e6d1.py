"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_transitive_successor_counts(times, predecessors):
    """Compute for each task the number of transitive successors (including direct)."""
    n = len(times)
    # Build successor adjacency
    succ = [set() for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if predecessors[i] is not None else []
        for p in preds:
            succ[p].add(i)
    
    # Compute transitive closure counts via DFS/BFS from each node
    # For efficiency with n=30, we can do a simple transitive closure
    transitive_succ = [set() for _ in range(n)]
    for i in range(n):
        # BFS/DFS to find all transitive successors
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(succ[node])
        transitive_succ[i] = visited
    
    counts = np.array([len(transitive_succ[i]) for i in range(n)], dtype=float)
    return counts


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    # Compute transitive successor counts (bottleneck indicator)
    trans_succ_counts = _compute_transitive_successor_counts(times, predecessors)
    
    # Normalize transitive successor counts to [0, 1]
    max_ts = trans_succ_counts.max() if n > 0 else 1.0
    if max_ts > 0:
        ts_norm = trans_succ_counts / max_ts
    else:
        ts_norm = np.zeros(n)
    
    # Base score: task time (longer tasks get higher priority)
    base = times.copy()
    
    # Add a bonus for tasks with many transitive successors
    # This breaks ties in favor of bottleneck tasks, unlocking downstream fan-out earlier
    bonus_weight = 0.5
    score = base + bonus_weight * ts_norm * times.max()
    
    return score
# EVOLVE-BLOCK-END
