import numpy as np
from collections import deque
import time

def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    if n == 0:
        return np.array([], dtype=float)

    # Extract predecessors
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        if j not in succs[i]:
                            succs[i].append(j)

    # Extract resource data
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break

    demands_arr = None
    capacities_arr = None
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(-1, 1)
        if capacities_arr.ndim == 0:
            capacities_arr = capacities_arr.reshape(1)
        if demands_arr.shape[0] != n:
            demands_arr = None
            capacities_arr = None
        elif capacities_arr.ndim == 1 and capacities_arr.size != demands_arr.shape[1]:
            # Try to make shapes compatible
            if capacities_arr.size == 1:
                capacities_arr = np.tile(capacities_arr, (demands_arr.shape[1],))
            elif demands_arr.shape[1] == 1:
                demands_arr = np.tile(demands_arr, (1, capacities_arr.size))
            else:
                demands_arr = None
                capacities_arr = None
        if demands_arr is not None:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)

    # Forward pass for ES
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]

    project_duration = float(np.max(ES + durations)) if n > 0 else 0.0

    # Backward pass for LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    total_float = LS - ES

    # Simulate serial SGS with different priority rules to get makespans
    # and track which activities are "critical" (on the resource-constrained critical path)

    def simulate_sgs(prio_array):
        """
        prio_array: numpy array of priorities (higher = scheduled earlier).
        Returns (makespan, start_times)
        """
        scheduled = [False] * n
        start_times = np.full(n, -1.0)
        current_time = 0.0
        remaining_in_degree = in_degree.copy()
        eligible = [i for i in range(n) if remaining_in_degree[i] == 0]

        # Track resource usage: for each resource, list of (end_time, amount)
        if demands_arr is not None:
            n_res = demands_arr.shape[1]
            # Use sorted lists of (end_time, amount) for efficient cleanup
            resource_usage = [[] for _ in range(n_res)]
        else:
            n_res = 0
            resource_usage = []

        def get_current_usage(r, t):
            """Return current resource usage at time t for resource r."""
            # Remove finished resources
            ru = resource_usage[r]
            while ru and ru[0][0] <= t:
                ru.pop(0)
            return sum(amt for _, amt in ru)

        while sum(remaining_in_degree) > 0:
            # Find eligible activities that can be started at current_time
            feasible = []
            for act in eligible:
                if scheduled[act]:
                    continue
                if demands_arr is not None:
                    ok = True
                    for r in range(n_res):
                        cur_use = get_current_usage(r, current_time)
                        if cur_use + demands_arr[act, r] > capacities_arr[r]:
                            ok = False
                            break
                    if not ok:
                        continue
                feasible.append(act)

            if not feasible:
                # Advance time to next event
                next_time = float('inf')
                if demands_arr is not None:
                    for r in range(n_res):
                        for end_t, amt in resource_usage[r]:
                            if end_t > current_time and end_t < next_time:
                                next_time = end_t
                if next_time == float('inf'):
                    break
                current_time = next_time
                continue

            # Pick the highest priority feasible activity
            best_act = max(feasible, key=lambda a: prio_array[a])
            scheduled[best_act] = True
            start_times[best_act] = current_time
            end_time = current_time + durations[best_act]

            if demands_arr is not None:
                for r in range(n_res):
                    if demands_arr[best_act, r] > 0:
                        resource_usage[r].append((end_time, demands_arr[best_act, r]))

            # Update successors
            for v in succs[best_act]:
                remaining_in_degree[v] -= 1
                if remaining_in_degree[v] == 0:
                    if v not in eligible:
                        eligible.append(v)

            # Advance time to end of this activity
            current_time = end_time
            if demands_arr is not None:
                for r in range(n_res):
                    while resource_usage[r] and resource_usage[r][0] <= current_time:
                        resource_usage[r].pop(0)

        # Compute makespan
        makespan = 0.0
        for i in range(n):
            if start_times[i] >= 0:
                makespan = max(makespan, start_times[i] + durations[i])

        return makespan, start_times

    # Define several priority rules as arrays
    def make_rule_array(scores):
        mn, mx = float(np.min(scores)), float(np.max(scores))
        if mx - mn < 1e-12:
            return np.ones(n, dtype=float)
        return (scores - mn) / (mx - mn)

    # Rule 1: Critical float (lower float = higher priority)
    rule1 = make_rule_array(-total_float)

    # Rule 2: LPT (longer duration = higher priority)
    rule2 = make_rule_array(durations)

    # Rule 3: Min float ratio
    float_ratio = np.where(total_float > 0, durations / (total_float + 1.0), durations * 10)
    rule3 = make_rule_array(float_ratio)

    # Rule 4: Resource heavy
    if demands_arr is not None:
        res_heavy = np.sum(demands_arr / capacities_arr, axis=1)
    else:
        res_heavy = durations.copy()
    rule4 = make_rule_array(res_heavy)

    # Rule 5: Combined
    if demands_arr is not None:
        rule5_raw = 0.5 * (-total_float) + 0.3 * durations + 0.2 * np.sum(demands_arr / capacities_arr, axis=1)
    else:
        rule5_raw = 0.7 * (-total_float) + 0.3 * durations
    rule5 = make_rule_array(rule5_raw)

    # Rule 6: Successor count (more successors = more impact)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    rule6 = make_rule_array(succ_count)

    # Rule 7: Late start (higher LS = higher priority, meaning less slack)
    rule7 = make_rule_array(-LS)

    all_rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7]

    # Simulate with each rule
    sim_results = []
    for norm_rule in all_rules:
        if time.time() - t0 > 100.0:
            break
        try:
            makespan, start_times = simulate_sgs(norm_rule)
            sim_results.append((makespan, start_times))
        except Exception:
            pass

    # Now compute a combined priority:
    # An activity gets high priority if:
    # 1. It has low total float (on critical path)
    # 2. It has long duration (LPT)
    # 3. It consumes scarce resources
    # 4. In the best simulation, it starts early (is part of the critical schedule)

    # Score from total float
    tf_score = -total_float
    tf_min, tf_max = float(np.min(tf_score)), float(np.max(tf_score))
    if tf_max - tf_min > 1e-12:
        tf_norm = (tf_score - tf_min) / (tf_max - tf_min)
    else:
        tf_norm = np.zeros(n)

    # Score from duration
    d_score = durations.copy()
    d_min, d_max = float(np.min(d_score)), float(np.max(d_score))
    if d_max - d_min > 1e-12:
        d_norm = (d_score - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros(n)

    # Score from resource usage
    if demands_arr is not None:
        r_score = np.sum(demands_arr / capacities_arr, axis=1)
        r_min, r_max = float(np.min(r_score)), float(np.max(r_score))
        if r_max - r_min > 1e-12:
            r_norm = (r_score - r_min) / (r_max - r_min)
        else:
            r_norm = np.zeros(n)
    else:
        r_norm = np.zeros(n)

    # Score from successor count (activities with more successors are more impactful)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    sc_min, sc_max = float(np.min(succ_count)), float(np.max(succ_count))
    if sc_max - sc_min > 1e-12:
        sc_norm = (succ_count - sc_min) / (sc_max - sc_min)
    else:
        sc_norm = np.zeros(n)

    # Score from simulation: in the best schedule, which activities start early?
    if sim_results:
        best_idx = int(np.argmin([r[0] for r in sim_results]))
        best_start = sim_results[best_idx][1]
        # Normalize start times: earlier start = higher priority
        valid_starts = best_start[best_start >= 0]
        if len(valid_starts) > 0:
            s_min, s_max = float(np.min(valid_starts)), float(np.max(valid_starts))
            if s_max - s_min > 1e-12:
                s_norm = 1.0 - (np.where(best_start >= 0, best_start, s_max) - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros(n)
        else:
            s_norm = np.zeros(n)
    else:
        s_norm = np.zeros(n)

    # Also consider: in all simulations, does this activity tend to be scheduled early?
    if len(sim_results) >= 2:
        avg_start = np.zeros(n)
        count = 0
        for _, st in sim_results:
            valid = st >= 0
            if np.any(valid):
                avg_start += np.where(valid, st, 0)
                count += 1
        if count > 0:
            avg_start /= count
            valid_avg = avg_start
            a_min, a_max = float(np.min(valid_avg)), float(np.max(valid_avg))
            if a_max - a_min > 1e-12:
                avg_s_norm = 1.0 - (valid_avg - a_min) / (a_max - a_min)
            else:
                avg_s_norm = np.zeros(n)
        else:
            avg_s_norm = np.zeros(n)
    else:
        avg_s_norm = s_norm.copy()

    # Combined score
    # Weight: critical path dominance
    w_tf = 0.30
    w_d = 0.15
    w_r = 0.20 if demands_arr is not None else 0.0
    w_s = 0.15
    w_as = 0.10
    w_sc = 0.10
    if demands_arr is None:
        w_tf = 0.35
        w_d = 0.20
        w_s = 0.15
        w_as = 0.10
        w_sc = 0.10

    total_w = w_tf + w_d + w_r + w_s + w_as + w_sc
    priority_scores = (w_tf * tf_norm + w_d * d_norm + w_r * r_norm +
                       w_s * s_norm + w_as * avg_s_norm + w_sc * sc_norm) / total_w

    # Add a small tiebreaker: lower index for stability
    priority_scores += np.arange(n) * 1e-10

    if time.time() - t0 > 110.0:
        return priority_scores

    return priority_scores
