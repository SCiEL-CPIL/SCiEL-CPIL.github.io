"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def priority(instance):
    """
    Returns a priority score for each activity for use in a Serial SGS.
    Uses a rank-based combination of critical path length, resource pressure,
    slack, and successor count to determine priority.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin below 120s limit

    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    N = len(durations)

    if "demands" in instance:
        demands = np.asarray(instance["demands"], dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(N, 1)
    else:
        demands = np.zeros((N, 1))

    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        capacities = np.ones(demands.shape[1])

    if "predecessors" in instance:
        preds = instance["predecessors"]
    elif "predecessor" in instance:
        preds = instance["predecessor"]
    else:
        preds = [list(range(i)) for i in range(N)]

    # Build adjacency and reverse adjacency
    successors = defaultdict(list)
    in_degree = np.zeros(N, dtype=int)
    for i in range(N):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological Sort and Forward Pass (EST)
    est = np.zeros(N, dtype=float)
    topological_order = []
    queue = list(np.where(in_degree == 0)[0])
    in_degree_copy = in_degree.copy()
    while queue:
        node = queue.pop(0)
        topological_order.append(node)
        for s in successors[node]:
            in_degree_copy[s] -= 1
            est[s] = max(est[s], est[node] + durations[node])
            if in_degree_copy[s] == 0:
                queue.append(s)

    # Backward Pass (LFT, LST)
    project_end = max(est[i] + durations[i] for i in range(N))
    lft = np.full(N, project_end, dtype=float)
    for node in reversed(topological_order):
        for s in successors[node]:
            lft[node] = min(lft[node], lft[s] - durations[s])
    
    slack = lft - est - durations

    # Calculate LPE (Longest Path to End)
    lpe = np.zeros(N, dtype=float)
    for node in reversed(topological_order):
        max_succ_lpe = 0.0
        for s in successors[node]:
            max_succ_lpe = max(max_succ_lpe, lpe[s])
        lpe[node] = durations[node] + max_succ_lpe

    # Resource Pressure
    if demands.shape[1] > 0:
        # Handle division by zero if capacity is 0 (shouldn't happen, but safe)
        safe_capacities = np.where(capacities == 0, 1e-9, capacities)
        resource_pressure = np.sum(demands / safe_capacities, axis=1)
    else:
        resource_pressure = np.zeros(N, dtype=float)

    # Successor Count
    succ_count = np.array([len(successors[i]) for i in range(N)], dtype=float)

    # Rank-based normalization
    def rank_normalize(arr):
        # Returns ranks from 1 (lowest) to N (highest)
        # We want higher value -> higher priority, so we rank ascending values as low priority
        # argsort gives indices that would sort the array.
        # We want the rank of each element.
        # If arr = [10, 20, 30], argsort = [0, 1, 2].
        # Rank of 10 is 1, 20 is 2, 30 is 3.
        # We want 30 to have high priority.
        # So we can just use the sorted position.
        sorter = np.argsort(arr)
        ranks = np.zeros_like(arr)
        ranks[sorter] = np.arange(1, len(arr) + 1)
        return ranks

    rank_lpe = rank_normalize(lpe)
    rank_pressure = rank_normalize(resource_pressure)
    rank_slack_inv = rank_normalize(-slack) # Low slack -> high rank
    rank_succ = rank_normalize(succ_count)
    
    # Weights
    # LPE is usually the strongest predictor.
    # Resource pressure is second.
    # Slack is correlated with LPE.
    # Successor count is a tie-breaker.
    
    w_lpe = 0.45
    w_pressure = 0.30
    w_slack = 0.15
    w_succ = 0.10

    priority_scores = (
        w_lpe * rank_lpe +
        w_pressure * rank_pressure +
        w_slack * rank_slack_inv +
        w_succ * rank_succ
    )

    return priority_scores
# EVOLVE-BLOCK-END
