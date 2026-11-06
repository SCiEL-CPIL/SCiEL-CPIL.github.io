"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We keep a deadline guard to avoid timeouts.
_DEADLINE = 150.0
_START_TIME = None


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme (SGS).
    The evaluator calls this to get a priority score per activity; higher score
    means higher priority (scheduled earlier when eligible).

    Strategy:
      1. Compute a static "critical path" length from each activity to the end
         (longest path through successors considering durations).
      2. Compute a dynamic congestion estimate: for each activity, estimate how
         much it competes with others for scarce resources. We approximate this
         by, for each resource, the sum over all activities of (demand_j / capacity_r)
         weighted by overlap likelihood (approximated by 1/(1 + |start_est_j - start_est_i|)).
         Since we don't have start times yet, we use a fluid approximation:
         the total demand on resource r divided by capacity gives a scarcity factor.
         Activities with high demand on scarce resources get a congestion penalty.
      3. Combine: priority = critical_path_length - congestion_penalty.
         Higher critical path = more important to schedule early.
         Higher congestion = should schedule early to free up resources sooner.

    Actually, let me reconsider. In serial SGS, we pick the highest-priority
    eligible activity. We want to minimize makespan.

    Key insight: activities that are on the critical path should be scheduled
    early. Also, activities that consume scarce resources should be scheduled
    early so that resources are freed up for others.

    Let's compute:
      - crit_len[i]: longest path from i to the end (including duration[i]).
      - For each resource r, scarcity_r = sum_j(demand[j][r]) / capacity[r].
      - congestion[i] = sum_r (demand[i][r] / capacity[r]) * scarcity_r
        This measures how much activity i contributes to congestion on scarce resources.

    Priority = crit_len[i] + alpha * congestion[i]

    Higher is better (scheduled earlier).
    """
    if _elapsed() > _DEADLINE - 5.0:
        # Safety: return a simple fallback
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = demands.shape[1] if demands.ndim == 2 else 1

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Compute critical path length from each node to the end (longest path including own duration)
    # Use topological order (reverse of predecessors)
    # Compute in-degree
    in_deg = [0] * n
    for i in range(n):
        in_deg[i] = len(predecessors[i])

    # Topological sort (Kahn's algorithm)
    from collections import deque
    q = deque()
    for i in range(n):
        if in_deg[i] == 0:
            q.append(i)

    topo_order = []
    in_deg_copy = in_deg[:]
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use durations
        return np.asarray(durations, dtype=float)

    # Compute longest path from each node to the end
    # crit[i] = durations[i] + max(crit[j] for j in successors[i]), or durations[i] if no successors
    crit = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            if crit[v] > max_succ:
                max_succ = crit[v]
        crit[u] = durations[u] + max_succ

    # Compute resource scarcity
    # For each resource r: total_demand[r] = sum_j demand[j][r]
    if demands.ndim == 1:
        demands_2d = demands.reshape(-1, 1)
    else:
        demands_2d = demands

    total_demand = np.sum(demands_2d, axis=0)  # shape (n_res,)
    # scarcity[r] = total_demand[r] / capacity[r], clipped to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scarcity = total_demand / capacities
    scarcity = np.where(np.isfinite(scarcity), scarcity, 0.0)

    # Congestion for each activity: sum over resources of (demand[i][r] / capacity[r]) * scarcity[r]
    # demand_over_cap[i, r] = demand[i][r] / capacity[r]
    with np.errstate(divide='ignore', invalid='ignore'):
        demand_over_cap = demands_2d / capacities[np.newaxis, :]
    demand_over_cap = np.where(np.isfinite(demand_over_cap), demand_over_cap, 0.0)

    # congestion[i] = sum_r demand_over_cap[i, r] * scarcity[r]
    congestion = np.dot(demand_over_cap, scarcity)  # shape (n,)

    # Combine: priority = crit[i] + alpha * congestion[i]
    # alpha: weight for congestion. Let's use alpha = 1.0 as a starting point.
    alpha = 1.0
    priority_scores = crit + alpha * congestion

    return np.asarray(priority_scores, dtype=float)
# EVOLVE-BLOCK-END
