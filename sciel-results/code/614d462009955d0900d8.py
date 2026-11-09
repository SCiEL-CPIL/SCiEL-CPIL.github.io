import numpy as np


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n_activities, n_resources)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (n_resources,)
    predecessors = instance["predecessors"]  # list of lists, predecessors[i] = list of pred indices

    n = len(durations)
    n_res = len(capacities)

    if n == 0:
        return np.array([])

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Topological sort (Kahn's algorithm) to get proper ordering
    in_degree = [len(predecessors[i]) for i in range(n)]
    topo_order = []
    queue = [i for i in range(n) if in_degree[i] == 0]
    while queue:
        # Pick node with highest priority among available (use simple index for now)
        node = queue.pop(0)
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Ensure all nodes are included (in case of cycles, though shouldn't happen)
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # Forward pass: earliest start/finish in topological order
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in topo_order:
        if predecessors[i]:
            es[i] = max(ef[p] for p in predecessors[i])
        ef[i] = es[i] + durations[i]

    # Project makespan (no resources)
    makespan = max(ef) if n > 0 else 0.0

    # Backward pass: latest start/finish in reverse topological order
    lf = np.full(n, makespan)
    ls = np.full(n, makespan)
    for i in reversed(topo_order):
        if successors[i]:
            lf[i] = min(ls[s] for s in successors[i])
        else:
            lf[i] = makespan
        ls[i] = lf[i] - durations[i]

    # Total float
    total_float = lf - ef

    # Resource pressure: sum of demand/capacity for each activity
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    resource_pressure = np.sum(demands / safe_cap[np.newaxis, :], axis=1)

    # Number of immediate successors
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Number of immediate predecessors
    pred_count = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    # Duration
    dur = durations

    # Normalize each component to [0, 1]
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Invert total_float: low float = high criticality = high priority
    crit_score = 1.0 - normalize(total_float)
    res_score = normalize(resource_pressure)
    succ_score = normalize(succ_count)
    pred_score = normalize(pred_count)
    dur_score = normalize(dur)

    # Composite priority: weighted sum
    # Emphasize criticality most, then resource pressure, then successor count,
    # with smaller weights for predecessor count and duration
    priority_score = 0.45 * crit_score + 0.25 * res_score + 0.15 * succ_score + 0.05 * pred_score + 0.10 * dur_score

    return priority_score
