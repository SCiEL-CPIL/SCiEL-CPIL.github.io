import numpy as np


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = len(capacities)

    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Topological Sort to ensure correct order for forward/backward passes
    # Using Kahn's algorithm
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Pop from end for stable ordering if needed, but order doesn't matter for CPM math
        u = queue.pop()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If graph is not DAG, topo_order might be incomplete, but RCPSP assumes DAG.
    # Reverse topo order for backward pass
    rev_topo_order = topo_order[::-1]

    # 1. Forward Critical Path Length (C_i)
    # C_i = duration[i] + max(C_s for s in successors[i])
    # This represents the length of the longest path starting from i.
    C = np.zeros(n)
    for i in rev_topo_order:
        if successors[i]:
            max_succ_C = max(C[s] for s in successors[i])
        else:
            max_succ_C = 0.0
        C[i] = durations[i] + max_succ_C

    # 2. Resource Pressure
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    resource_pressure = np.sum(demands / safe_cap[np.newaxis, :], axis=1)

    # 3. Optional: Duration itself? C_i already includes it.
    # 4. Optional: Successor Count?
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    # Weights: Emphasize Forward Criticality (C_i) most.
    # High C_i means "this activity starts a long chain", so schedule it early.
    crit_score = normalize(C)
    res_score = normalize(resource_pressure)
    
    # Let's try a mix. 0.8 Crit, 0.2 Res.
    priority_score = 0.8 * crit_score + 0.2 * res_score

    return priority_score
