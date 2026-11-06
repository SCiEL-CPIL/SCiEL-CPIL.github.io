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

    # 6. Compute cumulative resource demand with soft-max contention
    #    This captures the "resource pressure" that gets unlocked when i is scheduled.
    #    We use a log-sum-exp based approach to smoothly approximate the max
    #    contention across resources, which is more robust than simple summation.
    cumulative_resource_load = np.zeros(n)
    resource_bottleneck = np.zeros(n)  # New: bottleneck pressure component
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 2 and capacities_arr.ndim == 1:
            R = len(capacities_arr)
            safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
            # Per-activity, per-resource normalized demand
            norm_demands = demands_arr / safe_caps[np.newaxis, :]  # (n, R)
            
            # Compute cumulative resource load via reverse topological accumulation
            # For each activity, we track the total downstream resource consumption
            cumulative_resource_load = np.zeros(n)
            # Also track the "bottleneck" - the maximum normalized demand across
            # resources in the downstream subgraph, smoothed via log-sum-exp
            bottleneck_scores = np.zeros((n, R))  # (n, R) for log-sum-exp computation
            
            for i in reversed(topo_order):
                # Own contribution
                own_load = np.sum(norm_demands[i] * L_end[i])
                own_bottleneck = norm_demands[i] * (L_end[i] / max(np.max(L_end), 1e-6))
                
                # Aggregate from successors
                for s in successors[i]:
                    cumulative_resource_load[i] += cumulative_resource_load[s]
                    bottleneck_scores[i] += bottleneck_scores[s]
                
                cumulative_resource_load[i] += own_load
                bottleneck_scores[i] += own_bottleneck
            
            # Convert bottleneck_scores to a smooth max approximation using log-sum-exp
            # This gives us a smooth surrogate for the max contention across resources
            # Scale factor controls the sharpness of the max approximation
            scale = 10.0
            # Add small epsilon to avoid log(0)
            bottleneck_scaled = bottleneck_scores * scale
            bottleneck_max_val = np.max(bottleneck_scaled, axis=1, keepdims=True)
            # Soft-max: log(sum(exp(x))) / scale, but we want the max-like value
            # Use: (1/scale) * log(sum(exp(scale * x)))
            # For numerical stability: (1/scale) * (max + log(sum(exp(scale * (x - max)))))
            with np.errstate(over='ignore'):
                exp_vals = np.exp(bottleneck_scaled - bottleneck_max_val)
                log_sum_exp = bottleneck_max_val + np.log(np.sum(exp_vals, axis=1, keepdims=True))
                resource_bottleneck = (log_sum_exp / scale).flatten()
            
            # Normalize resource_bottleneck to [0,1]
            rb_min, rb_max = np.min(resource_bottleneck), np.max(resource_bottleneck)
            if rb_max > rb_min:
                resource_bottleneck = (resource_bottleneck - rb_min) / (rb_max - rb_min)
            else:
                resource_bottleneck = np.ones(n) * 0.5
    else:
        # No resource data; fall back to criticality-based
        cumulative_resource_load = L_end.copy()
        resource_bottleneck = np.ones(n) * 0.5

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
    #    - Resource bottleneck: smooth max of resource contention (new)
    #    - Descendant count: prioritizes activities that unblock many others
    
    # Normalize each component to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5

    crit_norm = normalize(criticality)
    load_norm = normalize(cumulative_resource_load)
    bottleneck_norm = resource_bottleneck  # Already normalized
    desc_norm = normalize(descendant_count)

    # Structural change: use a multiplicative interaction between criticality
    # and resource load, rather than a simple weighted sum. This rewards
    # activities that are BOTH on critical paths AND have high downstream
    # resource pressure, which directly targets resource contention on
    # critical paths.
    interaction = crit_norm * load_norm
    
    # New: interaction between criticality and bottleneck
    bottleneck_interaction = crit_norm * bottleneck_norm

    # Final score: blend of interaction terms and individual components
    # The interaction term gets the highest weight because it captures
    # the joint effect that drives makespan extension
    # Increased weight on bottleneck_interaction to capture resource contention
    scores = 0.35 * interaction + 0.20 * bottleneck_interaction + 0.20 * crit_norm + 0.10 * load_norm + 0.08 * bottleneck_norm + 0.07 * desc_norm

    return scores
