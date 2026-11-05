"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict, deque

def _compute_ranking(instance):
    """Compute a priority score for each task using a combination of
    task duration, position in the precedence DAG (rank), and number of
    successors. Higher score = higher priority."""
    times = np.asarray(instance["times"], dtype=float)
    preds = instance.get("predecessors", None)
    cycle_time = instance.get("cycle_time", 40.0)
    n = len(times)
    
    # Build adjacency
    if preds is None:
        preds = [[] for _ in range(n)]
    
    # Compute successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succs[i].append(j)
    
    # Compute rank (longest path from start to each node)
    # Using topological order
    in_degree = [len(preds[j]) for j in range(n)]
    topo_order = []
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(topo_order) != n:
        # Cycle detected, fall back to just times
        return times.copy()
    
    # Forward rank: longest path from any source to node
    rank = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            rank[v] = max(rank[v], rank[u] + times[u])
    
    # Backward rank: longest path from node to any sink
    # Process in reverse topological order
    back_rank = np.zeros(n)
    for u in reversed(topo_order):
        for v in succs[u]:
            back_rank[u] = max(back_rank[u], times[v] + back_rank[v])
    
    # Count successors (transitive)
    succ_count = np.zeros(n)
    for u in reversed(topo_order):
        for v in succs[u]:
            succ_count[u] = max(succ_count[u], 1 + succ_count[v])
    
    # Priority: combine duration, backward rank (how much work remains after this task),
    # and number of successors (tasks that depend on this one)
    # Higher backward rank means the task is earlier in the chain, so it should be placed first
    # But we want to prioritize tasks that have more "downstream" work to enable parallelism
    
    # Normalized scores
    t_max = times.max() if times.max() > 0 else 1.0
    br_max = back_rank.max() if back_rank.max() > 0 else 1.0
    sc_max = succ_count.max() if succ_count.max() > 0 else 1.0
    
    # Weighted combination
    # Emphasize tasks with long duration and many successors
    priority_scores = (
        0.4 * (times / t_max) +
        0.35 * (back_rank / br_max) +
        0.25 * (succ_count / sc_max)
    )
    
    return priority_scores

def priority(instance):
    """Score each assembly task. Stations are filled with the highest-priority
    eligible task that fits the cycle time."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    try:
        scores = _compute_ranking(instance)
    except Exception:
        times = np.asarray(instance["times"], dtype=float)
        return times
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
