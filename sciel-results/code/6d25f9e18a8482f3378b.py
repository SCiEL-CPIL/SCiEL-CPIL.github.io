"""Resource-Constrained Project Scheduling with dynamic resource-contention priority."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute priority scores for activities using a dynamic resource-contention
    combined with longest-path-to-end heuristic.
    
    The priority for each activity is:
      - Longest path from the activity to the end (backward pass), which captures
        downstream criticality.
      - Plus a resource-contention term: for each resource, the ratio of this
        activity's demand to capacity, weighted by how many other unscheduled
        activities compete for that resource. This forces high-demand tasks early.
    
    Higher priority = scheduled earlier.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    # demands: shape (n, num_resources) or (num_resources, n)
    demands_raw = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    if demands_raw is None or capacities is None:
        # Fallback: just use durations
        return durations.copy()

    demands = np.asarray(demands_raw, dtype=float)
    # Normalize demands to shape (n, R)
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
    elif demands.shape[0] != n:
        demands = demands.T

    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 1:
        R = len(capacities)
    else:
        R = capacities.shape[-1]
        capacities = capacities.reshape(-1)

    # Build successor lists
    successors = [[] for _ in range(n)]
    pred_sets = [set() for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            pred_sets[i].add(p)

    # Compute longest path to end (backward pass) using topological order
    # First, get topological order via Kahn's algorithm
    in_degree = [len(pred_sets[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    # If not all nodes in topo order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Backward pass: longest path from each node to any sink
    # lp[i] = durations[i] + max(lp[j] for j in successors[i], default=0)
    lp = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[i]:
            if lp[succ] > max_succ:
                max_succ = lp[succ]
        lp[i] = durations[i] + max_succ

    # Resource contention score:
    # For each activity i, for each resource r where demand[i,r] > 0:
    #   contention_r = demand[i,r] / capacities[r] * (number of other activities using resource r)
    # Sum over resources, take the max or sum.
    
    # Precompute for each resource, how many activities use it
    resource_user_count = np.zeros(R, dtype=float)
    for r in range(R):
        if r < demands.shape[1]:
            resource_user_count[r] = np.sum(demands[:, r] > 0)

    # For each activity, compute weighted contention
    # weight_r = demand[i,r] / capacities[r] * (resource_user_count[r] - 1) / max(1, resource_user_count[r])
    # This gives higher weight to resources that are more contested and where this activity uses more.
    
    contention = np.zeros(n, dtype=float)
    for i in range(n):
        c = 0.0
        for r in range(min(R, demands.shape[1])):
            d = demands[i, r]
            if d > 0 and capacities[r] > 0:
                ratio = d / capacities[r]
                # Contention: how much of this resource is used by others too
                others = resource_user_count[r] - 1
                if others > 0:
                    c += ratio * others
        contention[i] = c

    # Normalize contention to be comparable to lp
    max_lp = np.max(lp) if n > 0 else 1.0
    max_contention = np.max(contention) if n > 0 else 1.0

    # Combine: prioritize high lp (critical path) and high contention (resource bottleneck)
    # Scale contention so it contributes meaningfully
    if max_contention > 0 and max_lp > 0:
        # Blend: 0.6 * normalized_lp + 0.4 * normalized_contention
        norm_lp = lp / max_lp
        norm_contention = contention / max_contention
        scores = 0.6 * norm_lp + 0.4 * norm_contention
    else:
        scores = lp.copy()

    # Check deadline
    if time.time() - t_start > DEADLINE:
        return scores

    # Additional refinement: also factor in the activity's own duration
    # Longer activities on the critical path should be prioritized
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur > 0:
        norm_dur = durations / max_dur
        scores = 0.7 * scores + 0.3 * norm_dur

    return scores
# EVOLVE-BLOCK-END
