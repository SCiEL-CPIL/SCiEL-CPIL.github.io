"""New program for Resource-Constrained Project Scheduling (serial SGS).

Approach: Minimum Slack (Time Criticality) combined with Resource Intensity.
This uses a standard CPM-based heuristic where activities with the least
scheduling flexibility (lowest total float/slack) are prioritized. This is
distinct from the Bottom Level approach as it considers both forward and
backward paths relative to the project makespan, effectively targeting the
critical path first. Resource intensity is used as a tie-breaker or secondary
weight to handle resource constraints.
"""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Predecessor list per activity.
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    # Normalise predecessors to lists of ints.
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        if isinstance(p, (list, tuple, np.ndarray)):
            p = [int(x) for x in p]
        else:
            p = [int(p)]
        # Filter out self and out-of-range.
        p = [x for x in p if 0 <= x < n and x != i]
        pred_lists.append(p)

    # Successor list per activity.
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in pred_lists[i]:
            succ_lists[j].append(i)

    # ---- Topological Sort (Kahn's Algorithm) ----
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in pred_lists[i]:
            in_degree[i] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Handle potential cycles (should not happen in RCPSP)
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(list(remaining))

    # ---- Forward Pass: Earliest Start (ES) and Earliest Finish (EF) ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for i in topo_order:
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        else:
            ES[i] = 0.0
        EF[i] = ES[i] + durations[i]

    # Project Makespan (unconstrained)
    makespan = float(np.max(EF)) if n > 0 else 0.0

    # ---- Backward Pass: Latest Finish (LF) and Latest Start (LS) ----
    LF = np.zeros(n, dtype=float)
    LS = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        if succ_lists[i]:
            LF[i] = min(LS[j] for j in succ_lists[i])
        else:
            LF[i] = makespan
        LS[i] = LF[i] - durations[i]

    # ---- Total Float (Slack) ----
    Slack = LS - ES
    # Ensure non-negative slack (numerical stability)
    Slack = np.maximum(Slack, 0.0)

    # ---- Resource Intensity Calculation ----
    demands = np.asarray(instance.get("demands", None), dtype=float)
    capacities = np.asarray(instance.get("capacities", None), dtype=float)

    resource_intensity = np.zeros(n, dtype=float)
    if demands is not None and demands.size > 0:
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if demands.shape[0] != n:
            demands = np.zeros((n, 1), dtype=float)

        R = demands.shape[1]
        if capacities is None or np.asarray(capacities).size == 0:
            caps = np.ones(R, dtype=float)
        else:
            caps = np.asarray(capacities, dtype=float).reshape(-1)
            if caps.size == 1:
                caps = np.full(R, float(caps[0]), dtype=float)
            elif caps.size < R:
                tmp = np.ones(R, dtype=float)
                tmp[:caps.size] = caps
                caps = tmp
            else:
                caps = caps[:R]
            caps = np.maximum(caps, 1e-9)

        # Per-activity resource usage ratio
        ratio = demands / caps[np.newaxis, :]  # (n, R)
        
        # Simple intensity: sum of ratios
        resource_intensity = np.sum(ratio, axis=1)

        # Normalize intensity to [0, 1]
        ri_max = float(np.max(resource_intensity)) if n > 0 else 1.0
        if ri_max <= 1e-12:
            resource_intensity = np.zeros(n, dtype=float)
        else:
            resource_intensity = resource_intensity / ri_max

    # ---- Composite Priority ----
    # We want to prioritize low Slack.
    # Normalize Slack to [0, 1] where 0 is max slack, 1 is min slack (0).
    slack_max = float(np.max(Slack)) if n > 0 else 1.0
    if slack_max <= 1e-12:
        slack_norm = np.zeros(n, dtype=float)
    else:
        slack_norm = 1.0 - (Slack / slack_max)
        
    # Score = Slack_Urgency * (1 + Resource_Intensity)
    # This prioritizes critical path activities, with a boost for resource-heavy ones.
    score = slack_norm * (1.0 + resource_intensity)

    # Tie-breaker: Higher duration first (longer tasks on critical path are more impactful)
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max > 1e-12:
        dur_norm = durations / dur_max
        score += 0.01 * dur_norm

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
