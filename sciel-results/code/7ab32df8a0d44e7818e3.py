"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = np.asarray(instance["capacities"], dtype=float)
    n_resources = len(capacities)

    # Build successor lists and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    # If not all nodes are in topo_order, there's a cycle; fall back to 0..n-1
    if len(topo_order) != n:
        topo_order = list(range(n))

    # CPM Forward pass
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        if predecessors[i]:
            ES[i] = max(EF[p] for p in predecessors[i])
        EF[i] = ES[i] + durations[i]

    project_duration = max(EF)

    # CPM Backward pass (reverse topo order)
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    for i in reversed(topo_order):
        if successors[i]:
            LF[i] = min(LS[s] for s in successors[i])
        LS[i] = LF[i] - durations[i]

    # Number of immediate successors (Grade)
    num_successors = np.array([len(s) for s in successors], dtype=float)

    # Number of immediate predecessors
    num_predecessors = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    # Resource pressure: max over resources of demand/capacity
    safe_caps = np.where(capacities > 0, capacities, 1.0)
    demand_matrix = np.asarray(demands, dtype=float)
    # Handle case where demands might be 1D (single resource) or 2D
    if demand_matrix.ndim == 1:
        demand_matrix = demand_matrix.reshape(-1, 1)
        n_resources = 1
        safe_caps = safe_caps.reshape(-1, 1)
    pressure = np.max(demand_matrix / safe_caps, axis=1)

    # Total resource consumption (sum across resources, normalized by capacity)
    total_consumption = np.sum(demand_matrix / safe_caps, axis=1)

    # Normalize all components to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        if rng < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / rng

    norm_ls = normalize(LS)  # 0 = earliest (most urgent), 1 = latest
    norm_grade = normalize(num_successors)  # 0 = fewest successors, 1 = most
    norm_pred = normalize(num_predecessors)  # 0 = fewest predecessors, 1 = most
    norm_pressure = normalize(pressure)  # 0 = low pressure, 1 = high pressure
    norm_consumption = normalize(total_consumption)  # 0 = low, 1 = high

    # Score components:
    # 1. Low LS (criticality): higher score for lower LS
    # 2. Low Grade (MGR): higher score for fewer successors
    # 3. High Pressure: higher score for higher resource pressure
    # 4. High Predecessor count: activities with many predecessors are "deep" in the network
    # 5. High Total Consumption: activities that use many resources are more constraining

    # Weights tuned to balance temporal urgency with resource scarcity
    w_ls = 1.0
    w_grade = 0.3
    w_pressure = 0.6
    w_pred = 0.2
    w_consumption = 0.2

    priority_score = (
        w_ls * (1.0 - norm_ls) +
        w_grade * (1.0 - norm_grade) +
        w_pressure * norm_pressure +
        w_pred * norm_pred +
        w_consumption * norm_consumption
    )

    return priority_score
# EVOLVE-BLOCK-END
