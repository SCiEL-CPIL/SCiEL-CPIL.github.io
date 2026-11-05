"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


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
    from collections import deque
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

    # Latest Start Time (LS) is the primary indicator of urgency
    # Grade (number of immediate successors) is the secondary indicator
    # MGR (Minimum Grade Rule) suggests prioritizing activities with FEWER successors
    
    num_successors = np.array([len(s) for s in successors], dtype=float)

    # Normalize LS: Lower LS means higher priority.
    # We want higher score for lower LS.
    ls_min, ls_max = LS.min(), LS.max()
    ls_range = ls_max - ls_min + 1e-9
    norm_ls = (LS - ls_min) / ls_range  # 0 for earliest, 1 for latest
    
    # Normalize Grade: Lower Grade (fewer successors) means higher priority.
    succ_min, succ_max = num_successors.min(), num_successors.max()
    succ_range = succ_max - succ_min + 1e-9
    norm_grade = (num_successors - succ_min) / succ_range # 0 for fewest, 1 for most

    # Primary: Low LS (Criticality)
    # Secondary: Low Grade (MGR)
    # We invert both so that "good" properties (low LS, low Grade) contribute positively to the score.
    # score = w1 * (1 - norm_ls) + w2 * (1 - norm_grade)
    
    # Tuning weights: LS is generally more important, but MGR is a strong tiebreaker/secondary.
    w_ls = 1.0
    w_grade = 0.5
    
    priority_score = w_ls * (1.0 - norm_ls) + w_grade * (1.0 - norm_grade)

    return priority_score
# EVOLVE-BLOCK-END
