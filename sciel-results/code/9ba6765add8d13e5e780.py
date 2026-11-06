"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_graph(n, predecessors):
    """Build successor list and in-degree from predecessors (list of lists)."""
    succ = [[] for _ in range(n)]
    in_deg = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        for p in preds:
            succ[p].append(i)
            in_deg[i] += 1
    return succ, in_deg


def _downstream_weight(n, succ):
    """Total number of tasks in each task's successor subtree (including itself)."""
    weight = np.ones(n, dtype=float)
    # Process in reverse topological order (successors before predecessors)
    # Use a simple Kahn-like reverse: repeatedly pick nodes with no unprocessed successors
    processed = np.zeros(n, dtype=bool)
    remaining = n
    # To be safe with cycles, limit iterations
    order = []
    indeg_succ = np.zeros(n, dtype=int)  # number of unprocessed successors
    for i in range(n):
        indeg_succ[i] = len(succ[i])
    queue = [i for i in range(n) if indeg_succ[i] == 0]
    while queue and remaining > 0:
        # pick arbitrary
        i = queue.pop()
        if processed[i]:
            continue
        processed[i] = True
        remaining -= 1
        order.append(i)
        for p in range(n):
            if i in succ[p]:
                indeg_succ[p] -= 1
                if indeg_succ[p] == 0:
                    queue.append(p)
    # Fallback for any unprocessed (cycles): just use identity
    if remaining > 0:
        for i in range(n):
            if not processed[i]:
                order.append(i)
    # Now compute weights in reverse topological order (leaves first)
    for i in reversed(order):
        for s in succ[i]:
            weight[i] += weight[s]
    return weight


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


def _total_successor_count(n, succ):
    """Count total (transitive) successors for each node."""
    total = np.zeros(n, dtype=int)
    # Process in reverse topological order (leaves first)
    processed = np.zeros(n, dtype=bool)
    order = []
    indeg_succ = np.array([len(succ[i]) for i in range(n)], dtype=int)
    queue = [i for i in range(n) if indeg_succ[i] == 0]
    while queue:
        i = queue.pop()
        if processed[i]:
            continue
        processed[i] = True
        order.append(i)
        for p in range(n):
            if i in succ[p]:
                indeg_succ[p] -= 1
                if indeg_succ[p] == 0:
                    queue.append(p)
    if len(order) < n:
        for i in range(n):
            if not processed[i]:
                order.append(i)
    # Leaves first: total[i] = sum(1 + total[s] for s in succ[i])
    for i in reversed(order):
        total[i] = sum(1 + total[s] for s in succ[i])
    return total


def priority(instance):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = float(instance["cycle_time"])
    n = len(times)

    succ, in_deg = _build_graph(n, predecessors)

    # Downstream weight (total tasks in successor subtree including self)
    down_w = _downstream_weight(n, succ)

    # Total (transitive) successor count for the slack denominator
    total_succ = _total_successor_count(n, succ).astype(float)
    total_succ_safe = np.maximum(total_succ, 1.0)

    # Bottleneck slack using total successors: smaller ratio => higher priority
    eps = 1e-6
    slack = (cycle_time - times) / total_succ_safe
    slack_score = 1.0 / (slack + eps)

    # Downstream weight score
    down_score = np.log1p(down_w)

    # Predecessor count penalty
    pred_count = in_deg.astype(float)
    pred_penalty = 1.0 / (1.0 + 0.3 * pred_count)

    def _norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.ones_like(x)
        return (x - mn) / (mx - mn)

    s_down = _norm(down_score)
    s_slack = _norm(slack_score)
    s_pred = _norm(pred_penalty)
    s_time = _norm(times)

    # Base combination
    base = 1.0 * s_down + 0.7 * s_slack + 0.3 * s_pred + 0.2 * s_time

    # Local refinement: try a few weighting variants, keep the one with fewest stations
    best_base = base.copy()
    best_station_count = float('inf')

    if time.time() < deadline - 5.0:
        # Variant 1: current base
        st1, ass1, _ = _greedy_stations(n, times, succ, in_deg, base, cycle_time)
        best_station_count = st1
        best_base = base.copy()
        best_assigned = ass1.copy()

        if time.time() < deadline - 5.0:
            # Variant 2: heavier downstream, lighter slack
            base2 = 1.3 * s_down + 0.5 * s_slack + 0.2 * s_pred + 0.2 * s_time
            st2, ass2, _ = _greedy_stations(n, times, succ, in_deg, base2, cycle_time)
            if st2 < best_station_count:
                best_station_count = st2
                best_base = base2.copy()
                best_assigned = ass2.copy()

        if time.time() < deadline - 5.0:
            # Variant 3: heavier slack
            base3 = 0.8 * s_down + 1.0 * s_slack + 0.2 * s_pred + 0.2 * s_time
            st3, ass3, _ = _greedy_stations(n, times, succ, in_deg, base3, cycle_time)
            if st3 < best_station_count:
                best_station_count = st3
                best_base = base3.copy()
                best_assigned = ass3.copy()

        if time.time() < deadline - 5.0:
            # Variant 4: emphasize time (longest first bias)
            base4 = 0.7 * s_down + 0.5 * s_slack + 0.2 * s_pred + 0.5 * s_time
            st4, ass4, _ = _greedy_stations(n, times, succ, in_deg, base4, cycle_time)
            if st4 < best_station_count:
                best_station_count = st4
                best_base = base4.copy()
                best_assigned = ass4.copy()

    # Final tight-station boost on the best variant
    if time.time() < deadline - 3.0 and 'best_assigned' in dir():
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
