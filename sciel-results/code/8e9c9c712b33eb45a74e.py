"""RCPSP priority via optimized local search with fast SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def priority(instance):
    """
    Compute priority scores for RCPSP activities.
    Uses fast SGS simulation and local search to optimize priority ordering.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components ---
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists
    if succs is None:
        succs = [[] for _ in range(n)]
        for j in range(n):
            p = preds[j]
            if isinstance(p, (list, tuple, set, np.ndarray)):
                for i in p:
                    idx = int(i)
                    if 0 <= idx < n:
                        succs[idx].append(j)

    # Precompute pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_set[j])

    q = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        EF[u] = ES[u] + durations[u]
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LF = np.zeros(n)
    LS = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]

    slack = LS - ES

    # --- Feature computation ---
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    if max_slack > min_slack:
        norm_slack = (slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.zeros(n)

    # Resource criticality: how much of each resource's capacity is consumed
    # and how scarce the resource is
    res_criticality = np.zeros(n)
    for r in range(R):
        cap_r = capacities[r]
        if cap_r > 0:
            # Demand ratio for this resource
            demand_ratio = demands[:, r] / cap_r
            # Weight by scarcity: if total demand >> capacity, it's more critical
            total_demand = np.sum(demands[:, r])
            scarcity = total_demand / (cap_r * makespan_forward) if makespan_forward > 0 else 1.0
            res_criticality += demand_ratio * scarcity

    max_rc = np.max(res_criticality)
    if max_rc > 0:
        norm_rc = res_criticality / max_rc
    else:
        norm_rc = np.zeros(n)

    # Duration normalized
    max_dur = np.max(durations)
    if max_dur > 0:
        norm_dur = durations / max_dur
    else:
        norm_dur = np.zeros(n)

    # Successor count (breadth)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    max_sc = np.max(succ_count)
    if max_sc > 0:
        norm_sc = succ_count / max_sc
    else:
        norm_sc = np.zeros(n)

    # --- Fast SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Fast Serial SGS: O(N^2 * R) per activity scheduling.
        Uses event-based resource checking.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')

        start_t = np.full(n, -1.0)
        finish_t = np.full(n, -1.0)
        remaining = list(range(n))
        remaining_set = set(remaining)
        scheduled_count = 0

        # Track active activities for resource checking
        # We'll maintain a list of (start, finish, idx) for scheduled activities

        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf')

            # Find eligible activities (all predecessors scheduled)
            eligible = []
            for j in remaining:
                all_preds_done = True
                for p in pred_set[j]:
                    if start_t[p] < 0:
                        all_preds_done = False
                        break
                if all_preds_done:
                    eligible.append(j)

            if not eligible:
                # Fallback
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_set[best]:
                cand = start_t[p] + durations[p]
                if cand > es_best:
                    es_best = cand

            # Candidate times: es_best and finish times of scheduled activities >= es_best
            cand_times = [es_best]
            for k in range(n):
                if finish_t[k] >= 0 and finish_t[k] >= es_best:
                    cand_times.append(finish_t[k])

            cand_times = sorted(set(cand_times))

            best_start = es_best
            found = False

            for t in cand_times:
                if t < es_best:
                    continue
                end_t = t + durations[best]

                # Check resource feasibility using event points
                # Collect all event points in [t, end_t]
                event_points = {t, end_t}
                for k in range(n):
                    if start_t[k] >= 0:
                        if start_t[k] < end_t and finish_t[k] > t:
                            if t <= start_t[k] < end_t:
                                event_points.add(start_t[k])
                            if t < finish_t[k] <= end_t:
                                event_points.add(finish_t[k])

                sorted_events = sorted(event_points)

                feasible = True
                for i in range(len(sorted_events) - 1):
                    tp = sorted_events[i]
                    tp_next = sorted_events[i + 1]
                    if tp >= end_t or tp_next <= t:
                        continue

                    # Sum demands of activities active in [tp, tp_next)
                    # An activity k is active if start_t[k] <= tp and finish_t[k] >= tp_next
                    active_demand = np.zeros(R)
                    for k in range(n):
                        if start_t[k] >= 0:
                            if start_t[k] <= tp and finish_t[k] >= tp_next:
                                active_demand += demands[k]

                    active_demand += demands[best]

                    if np.any(active_demand > capacities + 1e-9):
                        feasible = False
                        break

                if feasible:
                    best_start = t
                    found = True
                    break

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            remaining.remove(best)
            remaining_set.discard(best)
            scheduled_count += 1

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Initial priority: combination of features ---
    # Weights to try
    weight_combos = [
        (1.0, 0.0, 0.0, 0.0),   # Pure slack
        (0.5, 0.5, 0.0, 0.0),   # Slack + Resource criticality
        (0.3, 0.4, 0.3, 0.0),   # Slack + RC + Duration
        (0.4, 0.3, 0.2, 0.1),   # All features
        (0.2, 0.6, 0.2, 0.0),   # RC dominant
        (0.5, 0.3, 0.2, 0.0),   # Slack dominant with RC
    ]

    best_scores = None
    best_ms = float('inf')

    for w1, w2, w3, w4 in weight_combos:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        scores = w1 * (1.0 - norm_slack) + w2 * norm_rc + w3 * norm_dur + w4 * norm_sc
        ms = simulate_sgs(scores)
        if ms < best_ms:
            best_ms = ms
            best_scores = scores.copy()

    if best_scores is None:
        best_scores = 1.0 - norm_slack
        best_ms = simulate_sgs(best_scores)

    # --- Local Search: Simulated Annealing ---
    current_scores = best_scores.copy()
    current_ms = best_ms

    T = 1.0
    T_min = 0.005
    cooling = 0.995

    # Time budget for search
    search_start = time.time()
    max_search_time = DEADLINE - (time.time() - start_time) - 3.0  # 3s safety

    iteration = 0
    while time.time() - search_start < max_search_time:
        if time.time() - start_time > DEADLINE:
            break

        # Generate neighbor
        r = random.random()
        if r < 0.4:
            # Swap two activities
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], neighbor_scores[i1]
        elif r < 0.7:
            # Perturb one activity
            i1 = random.randint(0, n - 1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.15)
        else:
            # Perturb two activities
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.1)
            neighbor_scores[i2] += random.gauss(0, 0.1)

        neighbor_ms = simulate_sgs(neighbor_scores)

        if neighbor_ms == float('inf'):
            continue

        delta = neighbor_ms - current_ms
        if delta < 0:
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            if T > T_min and random.random() < np.exp(-delta / T):
                current_scores = neighbor_scores
                current_ms = neighbor_ms

        T = max(T * cooling, T_min)
        iteration += 1

    return best_scores
# EVOLVE-BLOCK-END
