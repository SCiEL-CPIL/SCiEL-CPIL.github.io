"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score tasks by a combination of their own processing time and the number
    of transitive successors they have (downstream fan-out).
    
    Rationale:
    - High processing time: classic LPT heuristic, fills stations efficiently.
    - High successor count: tasks that unlock many downstream tasks should be
      scheduled early so the greedy packer has more eligible options later.
    
    We use a weighted sum: priority = time + alpha * (successor_count / max_successors)
    This keeps the scale of `time` dominant while adding a small boost for
    tasks that are bottlenecks for many successors.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    if n == 0:
        return np.array([], dtype=float)

    # Build successor graph
    # predecessors: list of lists, predecessors[i] = list of task indices that must come before i
    preds = instance["predecessors"]
    
    # Build adjacency: succ[i] = list of tasks that depend on i (i must come before succ[i])
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    
    # Compute transitive successor counts via BFS/DFS from each node
    # For 30 tasks, this is cheap.
    trans_succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = list(succ[i])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for nxt in succ[node]:
                    if nxt not in visited:
                        stack.append(nxt)
        trans_succ_count[i] = len(visited)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 5.0:
        # Fallback to simple time-based priority
        return times.copy()
    
    max_succ = trans_succ_count.max() if n > 0 else 1.0
    if max_succ == 0:
        return times.copy()
    
    # Normalize successor count to [0, 1]
    norm_succ = trans_succ_count / max_succ
    
    # Weight: keep time dominant, add a modest boost for high-fanout tasks
    # The boost should be small enough not to override time ordering significantly,
    # but large enough to break ties in favor of bottleneck tasks.
    alpha = 0.5  # Half a time-unit max boost
    
    scores = times + alpha * norm_succ
    
    return scores
# EVOLVE-BLOCK-END
