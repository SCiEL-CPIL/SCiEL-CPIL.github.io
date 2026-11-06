"""Rewritten program for Resource-Constrained Project Scheduling (serial SGS).

New approach family: CPM total-float (slack) based priority combined with
resource-scarcity pressure and most-preceding-successors (MPS). This is
fundamentally different from the seed's pure longest-duration-first ordering.
"""
# EVOLVE-BLOCK-START
import numpy as np


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

    # ---- CPM forward pass (earliest start/finish) ----
    ES = np.zeros(n, dtype=float)
    EF = np.zeros(n, dtype=float)
    for i in range(n):
        if pred_lists[i]:
            ES[i] = max(EF[j] for j in pred_lists[i])
        EF[i] = ES[i] + durations[i]
    project_duration = float(max(EF)) if n > 0 else 0.0

    # ---- CPM backward pass (latest start/finish) ----
    LF = np.full(n, project_duration, dtype=float)
    LS = np.full(n, project_duration, dtype=float)
    for i in range(n - 1, -1, -1):
        if succ_lists[i]:
            LF[i] = min(LS[j] for j in succ_lists[i])
        else:
            LF[i] = project_duration
        LS[i] = LF[i] - durations[i]

    # Total float (slack).
    slack = LS - ES  # == LF - EF
    # Guard against tiny negatives from float error.
    slack = np.maximum(slack, 0.0)

    # Normalise slack to [0, 1] (higher slack -> lower priority).
    slack_max = float(np.max(slack)) if n > 0 else 1.0
    if slack_max <= 1e-12:
        slack_norm = np.zeros(n, dtype=float)
    else:
        slack_norm = slack / slack_max  # 0 = critical, 1 = max float

    # ---- Resource scarcity pressure ----
    # demands: shape (n, R) or (n,) for single resource; capacities: (R,) or scalar.
    demands = np.asarray(instance.get("demands", None), dtype=float)
    capacities = np.asarray(instance.get("capacities", None), dtype=float)
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and demands.size > 0:
        if demands.ndim == 1:
            demands = demands.reshape(n, 1)
        if demands.shape[0] != n:
            # Fall back gracefully.
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

        # Per-activity usage ratio (sum over resources of demand/capacity).
        ratio = demands / caps[np.newaxis, :]  # (n, R)
        # Per-resource "tightness": how many activities need it, weighted by ratio.
        # A resource used by many activities with high ratio is scarce.
        # resource_load[r] = sum over activities of ratio[i, r]
        resource_load = np.sum(ratio, axis=0)  # (R,)
        load_max = float(np.max(resource_load)) if R > 0 else 1.0
        if load_max <= 1e-12:
            load_weight = np.ones(R, dtype=float)
        else:
            load_weight = resource_load / load_max  # (R,)

        # Activity pressure: weighted sum of its own ratios by resource tightness.
        resource_pressure = np.sum(ratio * load_weight[np.newaxis, :], axis=1)  # (n,)
        rp_max = float(np.max(resource_pressure)) if n > 0 else 1.0
        if rp_max <= 1e-12:
            resource_pressure = np.zeros(n, dtype=float)
        else:
            resource_pressure = resource_pressure / rp_max

    # ---- Most Preceding Successors (MPS) ----
    succ_counts = np.array([len(succ_lists[i]) for i in range(n)], dtype=float)
    sc_max = float(np.max(succ_counts)) if n > 0 else 1.0
    if sc_max <= 1e-12:
        mps = np.zeros(n, dtype=float)
    else:
        mps = succ_counts / sc_max

    # ---- Duration normalisation (small tie-breaker) ----
    dur_max = float(np.max(durations)) if n > 0 else 1.0
    if dur_max <= 1e-12:
        dur_norm = np.zeros(n, dtype=float)
    else:
        dur_norm = durations / dur_max

    # ---- Composite priority (higher = scheduled first) ----
    # Primary: low slack (critical path first).
    # Secondary: high resource pressure (scarce resources first).
    # Tertiary: many successors (unlock more work).
    # Quaternary: longer duration (break ties).
    w_slack = 1.0
    w_res = 0.5
    w_mps = 0.25
    w_dur = 0.1

    score = (
        -w_slack * slack_norm
        + w_res * resource_pressure
        + w_mps * mps
        + w_dur * dur_norm
    )

    return np.asarray(score, dtype=float)
# EVOLVE-BLOCK-END
