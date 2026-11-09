# EVOLVE-BLOCK-START
import time
import numpy as np

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).

    Combines:
      1. Critical-path distance to sink (longest path from activity to end).
      2. Resource bottleneck intensity: sum over resources of (demand/capacity)
         weighted by the number of direct successors (impact propagation).
      3. Duration as a tiebreaker / secondary signal.

    The resource term uses the idea that an activity with high demand relative
    to capacity AND many successors is the most constraining on the schedule.
    """
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Build predecessor / successor lists ---
    preds = instance.get("predecessors", None)
    if preds is None:
        # Try alternative keys
        for key in ("preds", "predecessor", "predecessor_list"):
            if key in instance:
                preds = instance[key]
                break
        if preds is None:
            preds = [[] for _ in range(n)]

    # Normalize preds to list of lists of ints
    if isinstance(preds, (list, tuple, np.ndarray)):
        if len(preds) == n:
            pred_lists = []
            for p in preds:
                if isinstance(p, (list, tuple, np.ndarray)):
                    pred_lists.append([int(x) for x in p if x is not None])
                elif p is None:
                    pred_lists.append([])
                else:
                    pred_lists.append([int(p)])
        else:
            pred_lists = [[] for _ in range(n)]
    else:
        pred_lists = [[] for _ in range(n)]

    # Build successor lists
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                succ_lists[p].append(i)

    # --- Compute longest path from each node to sink (critical path distance) ---
    # Use topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                in_degree[i] += 1

    # Topological sort
    from collections import deque
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If topo doesn't include all nodes (cycle), fall back to range
    if len(topo) < n:
        topo = list(range(n))

    # Longest path from node to sink: process in reverse topological order
    dist_to_sink = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if succ_lists[u]:
            dist_to_sink[u] = max(
                durations[u] + dist_to_sink[v] for v in succ_lists[u]
            )
        else:
            dist_to_sink[u] = durations[u]

    # --- Resource bottleneck intensity ---
    # demands: list/array of arrays, one per resource
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    if capacities is None:
        for key in ("resource_capacities", "caps", "capacity"):
            if key in instance:
                capacities = instance[key]
                break

    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)

        # demands shape: (n, R) or (R, n) — detect
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if capacities.ndim == 1:
            capacities = capacities.reshape(1, -1)

        # Ensure demands is (n, R)
        if demands.shape[0] != n and demands.shape[1] == n:
            demands = demands.T

        # capacities is (R,)
        if capacities.ndim == 2:
            capacities = capacities[0]

        R = demands.shape[1]

        # For each activity, compute sum over resources of (demand/capacity) * (1 + num_successors)
        # Higher = more constraining
        num_succ = np.array([len(s) for s in succ_lists], dtype=float)

        # demand_ratio: (n, R)
        # Avoid division by zero
        safe_caps = np.where(capacities > 0, capacities, 1.0)
        demand_ratio = demands / safe_caps[np.newaxis, :]  # (n, R)

        # Weight each resource's contribution by successor count (impact)
        # bottleneck_score[i] = sum_r demand_ratio[i,r] * (1 + num_succ[i])
        bottleneck_score = np.sum(demand_ratio, axis=1) * (1.0 + num_succ)

        # Normalize bottleneck to [0, 1] range for combination
        b_min, b_max = bottleneck_score.min(), bottleneck_score.max()
        if b_max > b_min:
            bottleneck_norm = (bottleneck_score - b_min) / (b_max - b_min)
        else:
            bottleneck_norm = np.zeros(n, dtype=float)
    else:
        bottleneck_norm = np.zeros(n, dtype=float)

    # --- Combine scores ---
    # Normalize critical path distance to [0, 1]
    d_min, d_max = dist_to_sink.min(), dist_to_sink.max()
    if d_max > d_min:
        crit_norm = (dist_to_sink - d_min) / (d_max - d_min)
    else:
        crit_norm = np.zeros(n, dtype=float)

    # Normalize duration to [0, 1]
    du_min, du_max = durations.min(), durations.max()
    if du_max > du_min:
        dur_norm = (durations - du_min) / (du_max - du_min)
    else:
        dur_norm = np.zeros(n, dtype=float)

    # Weighted combination: critical path is most important, then bottleneck, then duration
    score = 0.5 * crit_norm + 0.35 * bottleneck_norm + 0.15 * dur_norm

    # Time guard
    if time.time() - t0 > DEADLINE:
        return np.asarray(score, dtype=float)

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
