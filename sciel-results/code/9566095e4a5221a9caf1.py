# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 135.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demand"], dtype=float) if "demand" in instance else np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors and in-degree
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        preds = predecessors[j]
        if preds:
            in_degree[j] = len(preds)
            for p in preds:
                successors[p].append(j)

    # Topological sort (Kahn)
    queue = [j for j in range(n) if in_degree[j] == 0]
    top_order = []
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # If not all nodes reached (shouldn't happen), append the rest
    if len(top_order) < n:
        seen = set(top_order)
        for j in range(n):
            if j not in seen:
                top_order.append(j)

    rev_top_order = list(reversed(top_order))

    # 1) Critical path length from each node to end (longest remaining chain)
    cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            val = durations[j] + cp[j]
            if val > max_succ:
                max_succ = val
        cp[i] = durations[i] + max_succ

    # 1b) Smooth critical-path proxy via log-sum-exp over successor chains.
    #     This is a smooth surrogate for the nonsmooth max, giving a soft
    #     "width" of the critical region rather than a single hard path.
    LSE_T = 1.5  # temperature: smaller = closer to hard max
    soft_cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        vals = [durations[j] + soft_cp[j] for j in successors[i]]
        if vals:
            m = max(vals)
            if m > 1e-12:
                lse = m + LSE_T * np.log(np.sum(np.exp((np.array(vals) - m) / LSE_T)))
            else:
                lse = 0.0
        else:
            lse = 0.0
        soft_cp[i] = durations[i] + lse

    # 2) Transitive dependency count: number of nodes that transitively depend on i
    #    (i.e., size of the subproject that is blocked until i finishes).
    #    Computed by reverse topo: dep[i] = 1 + sum(dep[s] for s in succ(i))
    #    but that double counts shared descendants. For a cheap, monotone proxy
    #    of "unlocks a lot of work", use the count of direct+indirect successors
    #    via a union count is expensive; instead use longest-path *width* proxy:
    #    number of immediate successors is weak, so use the count of nodes in
    #    the forward cone via a boolean-union in reverse topo (n=30 is tiny).
    cone_count = np.zeros(n, dtype=int)
    cone_set = [None] * n
    for i in rev_top_order:
        s = set()
        for j in successors[i]:
            s.add(j)
            if cone_set[j] is not None:
                s |= cone_set[j]
        cone_set[i] = s
        cone_count[i] = len(s)

    max_cone = int(np.max(cone_count)) if n > 0 else 1
    if max_cone == 0:
        max_cone = 1
    norm_cone = cone_count.astype(float) / max_cone

    # 2b) Static slack: total float = cp[i] - (cp[i] - durations[i])
    #     Actually slack[i] = (latest_start - earliest_start). Since we only
    #     have forward CPM here, compute backward latest-finish for slack.
    #     latest_finish[i] = min over successors j of (latest_start[j])
    #     latest_start[j] = latest_finish[j] - durations[j]
    #     For terminal nodes, latest_finish = makespan (max cp)
    makespan_est = float(np.max(cp)) if n > 0 else 0.0
    latest_finish = np.full(n, makespan_est, dtype=float)
    for i in rev_top_order:
        if successors[i]:
            lf = np.inf
            for j in successors[i]:
                ls_j = latest_finish[j] - durations[j]
                if ls_j < lf:
                    lf = ls_j
            latest_finish[i] = lf
        # else: terminal, stays at makespan_est
    # total slack = latest_finish - (earliest_start + duration)
    # earliest_start[i] = max over preds of (est[p] + dur[p])
    earliest_start = np.zeros(n, dtype=float)
    for i in top_order:
        if predecessors[i]:
            es = 0.0
            for p in predecessors[i]:
                ef_p = earliest_start[p] + durations[p]
                if ef_p > es:
                    es = ef_p
            earliest_start[i] = es
    slack = latest_finish - (earliest_start + durations)
    max_slack = float(np.max(slack)) if n > 0 else 0.0
    if max_slack <= 0:
        max_slack = 1.0
    # Invert: low slack = high priority
    norm_slack_inv = 1.0 - slack / max_slack

    # 3) Resource pressure: bottleneck (max demand/capacity) as the dominant
    #    signal for "hard to schedule", plus a small sum term.
    n_res = len(capacities)
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    dur_weighted_res = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        dw = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
                dw += demands[j][k] * durations[j]
        max_ratio[j] = mx
        sum_ratio[j] = s
        dur_weighted_res[j] = dw

    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr

    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    max_dwr = float(np.max(dur_weighted_res)) if n > 0 else 0.0
    if max_dwr == 0:
        max_dwr = 1.0
    norm_dwr = dur_weighted_res / max_dwr

    # 3b) Resource-weighted tail: CPM tail where successor durations are
    #     weighted by their max resource ratio. Captains chains of resource-
    #     heavy activities that will contend for capacity.
    rw_tail = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_rw = 0.0
        for j in successors[i]:
            val = durations[j] * (1.0 + 0.5 * max_ratio[j]) + rw_tail[j]
            if val > max_rw:
                max_rw = val
        rw_tail[i] = durations[i] * (1.0 + 0.5 * max_ratio[i]) + max_rw

    max_rwt = float(np.max(rw_tail)) if n > 0 else 0.0
    if max_rwt == 0:
        max_rwt = 1.0
    norm_rwt = rw_tail / max_rwt

    # Normalize critical path
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine. Primary: critical path (longest remaining chain).
    # Secondary: resource-weighted tail (chains of resource-heavy activities).
    # Tertiary: transitive unlock (front-load gatekeepers).
    # Quaternary: resource bottleneck (hard-to-schedule activities first).
    # Quinary: low slack (critical-path proximity, smooth).
    # The soft_cp adds a smooth width signal alongside hard cp.
    max_scp = float(np.max(soft_cp)) if n > 0 else 0.0
    if max_scp == 0:
        max_scp = 1.0
    norm_soft_cp = soft_cp / max_scp

    w_cp = 1.0
    w_rwt = 0.35
    w_soft = 0.15
    w_cone = 0.25
    w_res = 0.15
    w_slack = 0.15
    score = (w_cp * norm_cp
             + w_rwt * norm_rwt
             + w_soft * norm_soft_cp
             + w_cone * norm_cone
             + w_res * (0.5 * norm_mr + 0.2 * norm_sr + 0.3 * norm_dwr)
             + w_slack * norm_slack_inv)

    return score
# EVOLVE-BLOCK-END
