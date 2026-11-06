"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    n = len(instance["durations"])
    preds = instance.get("predecessors", [])
    succs = [list() for _ in range(n)]
    indeg = np.zeros(n, dtype=int)
    for i, plist in enumerate(preds):
        for p in plist:
            succs[p].append(i)
            indeg[i] += 1
    return n, preds, succs, indeg


def _topo_order(n, preds, succs, indeg):
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    indeg_work = indeg.copy()
    while q:
        u = q.popleft()
        order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                q.append(v)
    if len(order) < n:
        # cycle fallback
        order = list(range(n))
    return order


def _forward_pass(n, durations, preds, order):
    es = np.zeros(n, dtype=float)
    for u in order:
        if preds[u]:
            es[u] = max(es[p] + durations[p] for p in preds[u])
    return es


def _backward_pass(n, durations, succs, order):
    ls = np.zeros(n, dtype=float)
    # project end
    project_end = max(es_last) if False else 0  # placeholder
    return ls, project_end


def priority(instance):
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin for 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [list() for _ in range(n)])

    # Ensure preds is a list of lists
    if not isinstance(preds, list):
        preds = [list(p) for p in preds]
    else:
        preds = [list(p) for p in preds]

    n, preds, succs, indeg = _build_graph(instance)
    order = _topo_order(n, preds, succs, indeg)

    # Forward pass: earliest start / earliest finish
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in order:
        if preds[u]:
            es[u] = max(es[p] + durations[p] for p in preds[u])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]

    # Project end (theoretical makespan without resources)
    project_end = float(max(ef)) if n > 0 else 0.0

    # Backward pass: latest finish / latest start
    lf = np.zeros(n, dtype=float)
    ls = np.zeros(n, dtype=float)
    # Initialize: activities with no successors have LF = project_end
    for i in range(n):
        if not succs[i]:
            lf[i] = project_end
    for u in reversed(order):
        if succs[u]:
            lf[u] = min(ls[v] for v in succs[u])
        ls[u] = lf[u] - durations[u]

    # Slack (total float)
    slack = ls - es  # >= 0 in a consistent DAG

    # Resource criticality: for each resource, compute how "tight" it is
    # We use a simple measure: sum of demands on the critical path vs capacity
    # But since we don't have a schedule yet, we approximate by:
    # For each activity, its "resource pressure" = sum over resources of (demand/capacity)
    # Weighted by how much that resource is contested (total demand / capacity)

    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)  # shape (n, R) or (n,)
        capacities = np.asarray(capacities, dtype=float)  # shape (R,) or scalar
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
            capacities = np.atleast_1d(capacities)
        R = demands.shape[1]
        # Total demand per resource across all activities
        total_demand = demands.sum(axis=0)  # shape (R,)
        # Contention ratio per resource
        contention = total_demand / np.maximum(capacities, 1e-9)  # shape (R,)
        # Per-activity resource pressure: weighted sum of demand/capacity
        # Use contention as weight to emphasize scarce resources
        resource_pressure = (demands / np.maximum(capacities, 1e-9)) @ contention

    # Normalize resource pressure to [0, 1] roughly
    max_rp = resource_pressure.max()
    if max_rp > 0:
        resource_pressure = resource_pressure / max_rp

    # Priority: combine inverse slack (criticality) with duration and resource pressure
    # Higher priority = should be scheduled earlier
    # Use: priority = duration + alpha * (1 - normalized_slack) + beta * resource_pressure
    # Normalize slack to [0, 1]
    max_slack = slack.max() if n > 0 else 1.0
    if max_slack > 0:
        norm_slack = slack / max_slack
    else:
        norm_slack = np.zeros(n, dtype=float)

    # Inverse slack: activities with less slack get higher priority
    inv_slack = 1.0 - norm_slack

    # Combine
    # Duration is already in time units; normalize it too for balance
    max_dur = durations.max() if n > 0 else 1.0
    if max_dur > 0:
        norm_dur = durations / max_dur
    else:
        norm_dur = np.zeros(n, dtype=float)

    # Weighted combination
    alpha = 0.5  # weight for criticality (inverse slack)
    beta = 0.3   # weight for resource pressure
    gamma = 0.2  # weight for duration

    priority_scores = gamma * norm_dur + alpha * inv_slack + beta * resource_pressure

    # Check time
    if time.time() > deadline - 5.0:
        return priority_scores

    # Return as a single score per activity
    return priority_scores
# EVOLVE-BLOCK-END
