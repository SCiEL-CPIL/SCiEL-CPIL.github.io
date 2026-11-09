import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.

    Design (refined within the CPM + resource-aware family):
    1. Compute Critical Path Length (CPL): longest path from each activity to project end.
       This is the primary, dominant priority signal.
    2. Compute a resource-contention signal using log-sum-exp smooth max over demand/capacity ratios.
       This favors activities that use scarce resources, reducing later blocking.
    3. Compute a resource-weighted tail: longest path where each node is weighted by its
       resource intensity * duration. This captures chains of resource-heavy activities.
    4. Compute slack (LST - EST) to penalize flexible activities.
    5. Small tie-break by duration for determinism.

    The serial SGS benefits most from scheduling critical-path activities first;
    the resource terms are secondary signals to avoid wasting scarce capacity and
    to prioritize activities that unblock resource-intensive downstream work.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n_res = len(capacities)

    # Build successor list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        # Fallback: just use durations as priority
        return durations.copy()

    # Compute Critical Path Length (CPL) - longest path from activity to project end
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute resource-contention signal using log-sum-exp smooth max over demand/capacity ratios.
    # This captures multi-resource pressure: activities with multiple moderately high ratios
    # get appropriately high scores (unlike hard max which ignores all but the single worst).
    tau = 0.08  # slightly smaller tau for sharper max approximation
    res_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        ratios = []
        for j in range(n_res):
            if capacities[j] > 0 and demands[i][j] > 0:
                ratios.append(demands[i][j] / capacities[j])
        if ratios:
            arr = np.asarray(rats if False else ratios, dtype=float)
            m = arr.max()
            if m > 1e-10:
                res_ratio[i] = tau * np.log(np.sum(np.exp((arr - m) / tau))) + m
            else:
                res_ratio[i] = 0.0
        else:
            res_ratio[i] = 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute resource-weighted tail: longest path where each node contributes
    # (resource_intensity * duration). This captures how much resource-heavy work
    # lies ahead of each activity, including its own contribution.
    res_tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        own_weight = res_ratio[u] * durations[u]
        if not successors[u]:
            res_tail[u] = own_weight
        else:
            max_succ = max(res_tail[v] for v in successors[u])
            res_tail[u] = own_weight + max_succ

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Latest Start Time (LST) to derive slack
    project_duration = max(cpl)
    lst = np.zeros(n, dtype=float)
    for u in topo_order:
        if not successors[u]:
            lst[u] = project_duration - durations[u]
        else:
            lst[u] = min(lst[v] - durations[u] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Earliest Start Time (EST) to derive slack
    est = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            est[u] = 0
        else:
            est[u] = max(est[p] + durations[p] for p in preds[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Total Slack = LST - EST (higher slack = more flexibility)
    slack = lst - est

    # Normalize components
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    cpl_norm = normalize(cpl)
    res_norm = normalize(res_ratio)
    tail_norm = normalize(res_tail)
    dur_norm = normalize(durations)
    slack_norm = normalize(slack)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl_norm.copy()

    # Combine scores:
    # Primary: Critical Path Length (dominant signal)
    # Secondary: Resource contention (boost for scarce-resource users)
    # Tertiary: Resource-weighted tail (downstream resource pressure)
    # Quaternary: Low slack (activities with less flexibility should be prioritized)
    # Quinary: Duration (small tie-break for determinism)
    #
    # The multiplicative coupling between CPL and resource ratio captures the idea
    # that a critical activity using scarce resources is especially urgent.
    # The resource tail adds a smooth signal for chains of resource-heavy work.
    # Negative slack_norm because lower slack should increase priority.
    scores = 1.0 * cpl_norm * (1.0 + 0.20 * res_norm) + 0.25 * tail_norm - 0.15 * slack_norm + 0.04 * dur_norm

    return scores
