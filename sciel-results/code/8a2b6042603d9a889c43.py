"""Resource-Constrained Project Scheduling: CPM-slack priority with smooth surrogate.

The serial SGS picks the highest-priority eligible activity. We rank activities so
that those with the least slack (i.e., on the critical path) and those that consume
scarce resources are scheduled first. We use a smooth (log-sum-exp) surrogate for
the longest downstream path to avoid nonsmoothness, and add auxiliary terms for
resource pressure and successor count.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_START = time.time()


def _remaining_time():
    return _DEADLINE - 5.0 - (time.time() - _START)  # 5s safety margin


def _build_successors(instance, n):
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    succ = [[] for _ in range(n)]
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        for j in p:
            j = int(j)
            if 0 <= j < n:
                succ[j].append(i)
    return succ


def _compute_priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # Demands and capacities for resource pressure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)

    succ = _build_successors(instance, n)

    # Forward pass: earliest start/finish (CPM, ignoring resources)
    es = np.zeros(n)
    ef = np.zeros(n)
    # Topological order via Kahn
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        p = instance.get("predecessors", None)
        if p is not None and p[i] is not None:
            for j in p[i]:
                j = int(j)
                if 0 <= j < n:
                    indeg[i] += 1
    from collections import deque
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # If cycle, fallback to index order
    if len(topo) != n:
        topo = list(range(n))

    for u in topo:
        for v in succ[u]:
            if es[v] < ef[u]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]

    # Backward pass: latest finish/start
    # Compute project end (max ef)
    project_end = float(ef.max()) if n > 0 else 0.0
    ls = np.full(n, project_end)
    lf = np.full(n, project_end)
    for u in reversed(topo):
        # lf[u] = min(lf[v] for v in succ[u]) or project_end if none
        if len(succ[u]) == 0:
            lf[u] = project_end
        else:
            lf[u] = float(min(lf[v] for v in succ[u]))
        ls[u] = lf[u] - durations[u]

    # Slack = LS - ES (non-negative ideally)
    slack = ls - es

    # Smooth surrogate: log-sum-exp of downstream path lengths as a soft criticality measure.
    # We use ef (earliest finish) as a proxy for how early the activity ends; activities
    # that end earlier but are on the critical path have low slack. We combine slack
    # with a smooth aggregation of successor durations to break ties smoothly.
    # Smooth downstream duration: for each activity, the log-sum-exp of (ef of successors)
    # scaled, which approximates the longest downstream path smoothly.
    # To keep it cheap and smooth, we compute a soft-max of successor ef values.
    soft_down = np.zeros(n)
    for u in range(n):
        if len(succ[u]) > 0:
            vals = np.array([ef[v] for v in succ[u]], dtype=float)
            m = vals.max()
            soft_down[u] = m + np.log(np.sum(np.exp(vals - m)))
        else:
            soft_down[u] = ef[u]

    # Normalize components to [0,1] for weighting
    def _norm(x):
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        if rng < 1e-12:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    slack_norm = _norm(slack)
    soft_down_norm = _norm(soft_down)
    dur_norm = _norm(durations)

    # Resource pressure: for each activity, sum over resources of demand/capacity
    res_pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        # demands shape: (n, R) or (n,) ; capacities shape: (R,) or (n, R)
        try:
            if demands.ndim == 1:
                d = demands.reshape(n, 1)
                c = np.asarray(capacities, dtype=float).reshape(-1, 1)
            else:
                d = demands
                c = capacities
            # Compute per-activity pressure = sum_r d_{i,r} / c_r
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = d / c
                ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            res_pressure = ratio.sum(axis=1)
        except Exception:
            res_pressure = np.zeros(n)
    res_pressure_norm = _norm(res_pressure)

    # Successor count (MPS-like: more successors => higher priority to unlock many)
    succ_count = np.array([len(succ[i]) for i in range(n)], dtype=float)
    succ_count_norm = _norm(succ_count)

    # Priority: we want higher score => scheduled first.
    # Low slack is good (critical path) => use (1 - slack_norm)
    # High soft_down (long downstream) is good => soft_down_norm
    # High resource pressure is good => res_pressure_norm
    # High successor count is good => succ_count_norm
    # Weighted combination; weights chosen to emphasize slack (critical path) most.
    w_slack = 0.50
    w_soft = 0.20
    w_res = 0.20
    w_succ = 0.10

    priority_scores = (
        w_slack * (1.0 - slack_norm)
        + w_soft * soft_down_norm
        + w_res * res_pressure_norm
        + w_succ * succ_count_norm
    )

    # Add a tiny duration tiebreaker (longer first) for determinism and stability
    priority_scores += 0.01 * dur_norm

    return priority_scores


def priority(instance):
    # Deadline guard: if we're close to the limit, return a safe fallback
    if _remaining_time() < 1.0:
        return np.asarray(instance["durations"], dtype=float)
    try:
        return _compute_priority(instance)
    except Exception:
        # Fallback to duration-based if anything goes wrong
        return np.asarray(instance["durations"], dtype=float)
# EVOLVE-BLOCK-END
