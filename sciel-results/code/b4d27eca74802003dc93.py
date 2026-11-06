import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]

    # Extract resource data if available
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    # 1. Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]

    while current_queue:
        node = current_queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)

    if len(topo_order) != n:
        # Cycle detected? Fallback to durations
        return durations.copy()

    # 3. Compute L_start (Longest path from start to i)
    L_start = np.zeros(n)
    for i in topo_order:
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred

    # 4. Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    for i in reversed(topo_order):
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ

    # 5. Criticality: total path length through activity i
    criticality = L_start + L_end - durations

    # 6. Compute cumulative resource demand: for each activity, sum of
    #    (demand * L_end) over all descendants including self.
    #    This captures the "resource load" that gets unlocked when i is scheduled.
    cumulative_resource_load = np.zeros(n)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 2 and capacities_arr.ndim == 1:
            # For each activity, compute a scalar "resource intensity"
            # weighted by the critical path length of its downstream subgraph
            # Use sum of (demand_r * L_end[i]) for all r, normalized by capacity
            safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
            # Per-activity, per-resource normalized demand
            norm_demands = demands_arr / safe_caps[np.newaxis, :]  # (n, R)
            # Weight by L_end to emphasize downstream resource consumption
            # cumulative_resource_load[i] = sum over all descendants d of i (including i) of sum_r norm_demands[d,r]
            # We compute this via reverse topological accumulation
            cumulative_resource_load = np.zeros(n)
            for i in reversed(topo_order):
                # Start with own weighted demand
                own_load = np.sum(norm_demands[i] * L_end[i])
                # Add contributions from successors
                for s in successors[i]:
                    # The load from successor s is already computed; we add it
                    # But we need to be careful: cumulative_resource_load[s] already includes s's own load
                    # We just propagate the total load from successors
                    cumulative_resource_load[i] += cumulative_resource_load[s]
                cumulative_resource_load[i] += own_load
    else:
        # No resource data; fall back to criticality-based
        cumulative_resource_load = L_end.copy()

    # 7. Successor count: how many direct successors does i have?
    #    More successors = more blocking potential
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # 8. Number of descendants (total, not just direct)
    #    Computed via reverse topological accumulation
    descendant_count = np.zeros(n)
    for i in reversed(topo_order):
        for s in successors[i]:
            descendant_count[i] += 1 + descendant_count[s]

    # 9. Combine into a single priority score:
    #    - Criticality: primary signal for makespan
    #    - Cumulative resource load: prioritizes activities that unlock heavy downstream resource use
    #    - Descendant count: prioritizes activities that unblock many others
    #    - Duration: longer tasks benefit from earlier start
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5

    crit_norm = normalize(criticality)
    load_norm = normalize(cumulative_resource_load)
    desc_norm = normalize(descendant_count)
    dur_norm = normalize(durations)

    # Structural change: use a multiplicative interaction between criticality
    # and resource load, rather than a simple weighted sum. This rewards
    # activities that are BOTH on critical paths AND have high downstream
    # resource pressure, which directly targets resource contention on
    # critical paths.
    interaction = crit_norm * load_norm

    # Final score: blend of interaction term and individual components
    # The interaction term gets the highest weight because it captures
    # the joint effect that drives makespan extension
    scores = 0.5 * interaction + 0.25 * crit_norm + 0.15 * load_norm + 0.10 * desc_norm

    return scores
