"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque


def _build_graph(n, predecessors):
    """Build successor list and in-degree from predecessors (list of lists)."""
    succ = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        for p in preds:
            succ[p].append(i)
            in_deg[i] += 1
    return succ, in_deg


def _topo_order(n, succ):
    """Return a topological order (predecessors before successors)."""
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        for s in succ[i]:
            in_deg[s] += 1
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in succ[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if len(order) < n:
        for i in range(n):
            if i not in set(order):
                order.append(i)
    return order


def _downstream_weight(n, succ):
    """Total number of tasks in each task's successor subtree (including itself)."""
    weight = np.ones(n, dtype=float)
    order = _topo_order(n, succ)
    for i in reversed(order):
        for s in succ[i]:
            weight[i] += weight[s]
    return weight


def _downstream_time(n, succ, times):
    """Total time in each task's successor subtree (including itself)."""
    weight = times.copy().astype(float)
    order = _topo_order(n, succ)
    for i in reversed(order):
        for s in succ[i]:
            weight[i] += weight[s]
    return weight


def _longest_path_to_end(n, succ, times):
    """Longest path from each node to any sink (inclusive of self)."""
    lp = times.copy().astype(float)
    order = _topo_order(n, succ)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if lp[s] > best:
                best = lp[s]
        lp[i] = times[i] + best
    return lp


def _total_successor_count(n, succ):
    """Count total (transitive) successors for each node."""
    total = np.zeros(n, dtype=int)
    order = _topo_order(n, succ)
    for i in reversed(order):
        total[i] = sum(1 + total[s] for s in succ[i])
    return total


def _greedy_stations(n, times, succ, in_deg, priority_scores, cycle_time):
    """Simulate the greedy station filling. Returns station index per task and total stations."""
    assigned = np.full(n, -1, dtype=int)
    station_load = 0.0
    station_idx = 0
    eligible = [i for i in range(n) if in_deg[i] == 0]
    remaining_indeg = in_deg.copy()
    placed = 0
    while placed < n:
        if not eligible:
            station_idx += 1
            station_load = 0.0
            continue
        best = None
        best_p = -np.inf
        for i in eligible:
            if times[i] <= cycle_time - station_load:
                if priority_scores[i] > best_p:
                    best_p = priority_scores[i]
                    best = i
        if best is None:
            station_idx += 1
            station_load = 0.0
            continue
        assigned[best] = station_idx
        station_load += times[best]
        placed += 1
        eligible.remove(best)
        for s in succ[best]:
            remaining_indeg[s] -= 1
            if remaining_indeg[s] == 0:
                eligible.append(s)
    return station_idx + 1, assigned, station_load


def priority(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)

    succ, in_deg = _build_graph(n, predecessors)

    # Feature 1: Downstream task count (log)
    down_w = _downstream_weight(n, succ)
    down_score = np.log1p(down_w)

    # Feature 2: Downstream time (log)
    down_t = _downstream_time(n, succ, times)
    down_t_score = np.log1p(down_t)

    # Feature 3: Longest path to end
    lp = _longest_path_to_end(n, succ, times)
    lp_score = lp

    # Feature 4: Bottleneck slack using total successors
    total_succ = _total_successor_count(n, succ).astype(float)
    total_succ_safe = np.maximum(total_succ, 1.0)
    eps = 1e-6
    slack = (cycle_time - times) / total_succ_safe
    slack_score = 1.0 / (slack + eps)

    # Feature 5: Immediate successor count
    imm_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    imm_succ_safe = np.maximum(imm_succ, 1.0)
    imm_slack = (cycle_time - times) / imm_succ_safe
    imm_slack_score = 1.0 / (imm_slack + eps)

    # Feature 6: Predecessor count penalty
    pred_count = in_deg.astype(float)
    pred_penalty = 1.0 / (1.0 + 0.3 * pred_count)

    # Feature 7: Task duration
    s_time_raw = times.copy()

    def _norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    s_down = _norm(down_score)
    s_down_t = _norm(down_t_score)
    s_lp = _norm(lp_score)
    s_slack = _norm(slack_score)
    s_imm_slack = _norm(imm_slack_score)
    s_pred = _norm(pred_penalty)
    s_time = _norm(s_time_raw)

    # Base combination: emphasize downstream volume and criticality
    base = (1.0 * s_down + 
            0.8 * s_down_t + 
            0.9 * s_lp + 
            0.7 * s_slack + 
            0.5 * s_imm_slack + 
            0.3 * s_pred + 
            0.2 * s_time)

    best_base = base.copy()
    best_station_count = float('inf')
    best_assigned = None

    # Local refinement: try several weighting variants
    variants = [
        # (name, weights tuple)
        ("base", (1.0, 0.8, 0.9, 0.7, 0.5, 0.3, 0.2)),
        ("down_heavy", (1.5, 1.2, 0.8, 0.5, 0.4, 0.2, 0.2)),
        ("slack_heavy", (0.8, 0.6, 0.7, 1.2, 1.0, 0.2, 0.2)),
        ("time_heavy", (0.7, 0.6, 0.7, 0.5, 0.4, 0.2, 0.6)),
        ("lp_heavy", (0.9, 0.7, 1.5, 0.6, 0.4, 0.2, 0.2)),
        ("balanced", (1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)),
    ]

    for name, w in variants:
        if time.time() >= deadline - 5.0:
            break
        v_base = (w[0] * s_down + w[1] * s_down_t + w[2] * s_lp +
                  w[3] * s_slack + w[4] * s_imm_slack + w[5] * s_pred +
                  w[6] * s_time)
        st, ass, _ = _greedy_stations(n, times, succ, in_deg, v_base, cycle_time)
        if st < best_station_count:
            best_station_count = st
            best_base = v_base.copy()
            best_assigned = ass.copy()

    # Final tight-station boost on the best variant
    if time.time() < deadline - 3.0 and best_assigned is not None:
        min_rem = {}
        for i in range(n):
            st = best_assigned[i]
            rem = cycle_time - times[i]
            if st not in min_rem or rem < min_rem[st]:
                min_rem[st] = rem
        tight = np.zeros(n, dtype=float)
        for i in range(n):
            st = best_assigned[i]
            tight[i] = 1.0 / (min_rem[st] + eps)
        tight = _norm(tight)
        best_base = best_base + 0.12 * tight * s_down

    best_base = np.nan_to_num(best_base, nan=0.0, posinf=0.0, neginf=0.0)
    return best_base
# EVOLVE-BLOCK-END
