"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def priority(instance):
    """
    Returns a priority score for each activity for use in a Serial SGS.
    Combines critical path length (longest path to end), resource pressure,
    and duration to produce a heuristic priority.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin below 120s limit

    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    N = len(durations)

    # Handle demands and capacities
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

    # Handle predecessors
    if "predecessors" in instance:
        preds = instance["predecessors"]
    elif "predecessor" in instance:
        preds = instance["predecessor"]
    else:
        preds = [list(range(i)) for i in range(N)]  # fallback: chain

    # Build adjacency and reverse adjacency
    # preds[i] is list of predecessors of i
    # We need successors for forward pass
    successors = defaultdict(list)
    in_degree = np.zeros(N, dtype=int)
    for i in range(N):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Forward pass: earliest start times (ignoring resources)
    # longest path from start to each node
    est = np.zeros(N, dtype=float)
    topological_order = []
    queue = list(np.where(in_degree == 0)[0])
    processed = np.zeros(N, dtype=bool)
    while queue:
        node = queue.pop(0)
        processed[node] = True
        topological_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            est[s] = max(est[s], est[node] + durations[node])
            if in_degree[s] == 0:
                queue.append(s)

    # Backward pass: latest finish times (ignoring resources)
    # longest path from each node to end
    # First compute the makespan (project duration without resource constraints)
    project_end = max(est[i] + durations[i] for i in range(N))

    # Reverse topological order
    reverse_topo = list(reversed(topological_order))
    lft = np.full(N, project_end, dtype=float)  # latest finish time
    for node in reverse_topo:
        for s in successors[node]:
            lft[node] = min(lft[node], lft[s] - durations[s])
        lft[node] = max(lft[node], est[node] + durations[node])  # ensure valid

    # Slack = lft - est - duration
    slack = lft - est - durations

    # Critical path length from node to end (including its own duration)
    # This is lft[node] - est[node] if we define it as the longest path through node
    # Actually, the longest path from node to end is:
    # Let's compute it properly
    path_to_end = np.zeros(N, dtype=float)
    for node in reverse_topo:
        max_successor_path = 0.0
        for s in successors[node]:
            max_successor_path = max(max_successor_path, path_to_end[s])
        path_to_end[node] = durations[node] + max_successor_path

    # Resource pressure: for each activity, sum of demand/capacity ratio
    # Higher pressure means more critical
    if demands.shape[1] > 0:
        resource_pressure = np.sum(demands / capacities, axis=1)
    else:
        resource_pressure = np.zeros(N, dtype=float)

    # Normalize components
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    norm_path = normalize(path_to_end)
    norm_slack_inv = normalize(-slack)  # lower slack = higher priority
    norm_pressure = normalize(resource_pressure)
    norm_duration = normalize(durations)

    # Combine with weights
    # Emphasize critical path and low slack
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    priority_scores = (
        weights[0] * norm_path +
        weights[1] * norm_slack_inv +
        weights[2] * norm_pressure +
        weights[3] * norm_duration
    )

    return priority_scores
# EVOLVE-BLOCK-END
