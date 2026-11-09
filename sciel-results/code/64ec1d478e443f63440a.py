# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_SECONDS = 135.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    if "demand" in instance:
        demands = np.asarray(instance["demand"], dtype=float)
    else:
        demands = np.asarray(instance["demands"], dtype=float)
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

    if len(top_order) < n:
        seen = set(top_order)
        for j in range(n):
            if j not in seen:
                top_order.append(j)

    rev_top_order = list(reversed(top_order))

    # 1) Forward Pass: Earliest Start (ES)
    es = np.zeros(n, dtype=float)
    for i in top_order:
        if predecessors[i]:
            max_ef = 0.0
            for p in predecessors[i]:
                ef_p = es[p] + durations[p]
                if ef_p > max_ef:
                    max_ef = ef_p
            es[i] = max_ef

    # 2) Backward Pass: Critical Path length (cp)
    cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ_cp = 0.0
        for j in successors[i]:
            if cp[j] > max_succ_cp:
                max_succ_cp = cp[j]
        cp[i] = durations[i] + max_succ_cp

    makespan_cpm = float(np.max(cp)) if n > 0 else 0.0
    if makespan_cpm == 0:
        makespan_cpm = 1.0

    # 3) Backward Pass: Latest Finish (LF) and Latest Start (LS)
    lf = np.full(n, makespan_cpm, dtype=float)
    for i in rev_top_order:
        if successors[i]:
            min_ls = np.inf
            for j in successors[i]:
                ls_j = lf[j] - durations[j]
                if ls_j < min_ls:
                    min_ls = ls_j
            lf[i] = min_ls

    ls = lf - durations
    slack = ls - es
    slack = np.maximum(slack, 0.0)

    max_slack = float(np.max(slack)) if n > 0 else 0.0
    if max_slack == 0:
        max_slack = 1.0
    norm_slack_inv = 1.0 - (slack / max_slack)

    # 4) Soft Critical Path (LSE) with adaptive temperature
    LSE_T = 1.5
    soft_cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        vals = [durations[j] + soft_cp[j] for j in successors[i]]
        if vals:
            m = max(vals)
            if m > 1e-12:
                exp_terms = np.exp((np.array(vals) - m) / LSE_T)
                lse = m + LSE_T * np.log(np.sum(exp_terms))
            else:
                lse = 0.0
        else:
            lse = 0.0
        soft_cp[i] = durations[i] + lse

    max_scp = float(np.max(soft_cp)) if n > 0 else 0.0
    if max_scp == 0:
        max_scp = 1.0
    norm_scp = soft_cp / max_scp

    # 5) Transitive Dependency Count (Cone)
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

    # 6) Resource Pressure - enhanced with duration-weighted impact
    n_res = len(capacities)
    LSE_T_RES = 0.4
    soft_max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    dur_weighted_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        ratios = []
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                ratios.append(r)
        if ratios:
            arr = np.array(ratios, dtype=float)
            m = float(np.max(arr))
            if m > 1e-12:
                exp_terms = np.exp((arr - m) / LSE_T_RES)
                soft_max_ratio[j] = m + LSE_T_RES * np.log(np.sum(exp_terms))
            else:
                soft_max_ratio[j] = 0.0
            sum_ratio[j] = float(np.sum(arr))
            dur_weighted_ratio[j] = durations[j] * float(np.max(arr))
        else:
            soft_max_ratio[j] = 0.0
            sum_ratio[j] = 0.0
            dur_weighted_ratio[j] = 0.0

    max_mr = float(np.max(soft_max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = soft_max_ratio / max_mr

    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    max_dw = float(np.max(dur_weighted_ratio)) if n > 0 else 0.0
    if max_dw == 0:
        max_dw = 1.0
    norm_dw = dur_weighted_ratio / max_dw

    # 7) Duration normalization
    max_dur = float(np.max(durations)) if n > 0 else 1.0
    if max_dur == 0:
        max_dur = 1.0
    norm_dur = durations / max_dur

    # Normalize Hard CP
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # 8) Bottleneck resource contention: for each activity, compute how much
    # it contributes to the tightest resource. Use a soft-min over resources
    # of (capacity - demand) to find the most constrained resource, then
    # weight by how much this activity consumes of it.
    # This captures "which resource will this activity stress most?"
    bottleneck_score = np.zeros(n, dtype=float)
    for j in range(n):
        if n_res > 0:
            remaining = capacities - demands[j]
            # Soft-min of remaining capacity (how tight is each resource)
            # Lower remaining = more constrained
            neg_remaining = -remaining
            m = float(np.max(neg_remaining))
            if m > -1e12:
                exp_terms = np.exp((neg_remaining - m) / 0.3)
                soft_min_remaining = -(m + 0.3 * np.log(np.sum(exp_terms)))
            else:
                soft_min_remaining = float(np.max(remaining))
            # The activity's demand on the most constrained resource
            max_demand = float(np.max(demands[j]))
            bottleneck_score[j] = max_demand / (soft_min_remaining + 1e-6)

    max_bn = float(np.max(bottleneck_score)) if n > 0 else 0.0
    if max_bn == 0:
        max_bn = 1.0
    norm_bn = bottleneck_score / max_bn

    # Combine Scores with tuned weights
    # Emphasize: critical path, resource bottleneck, soft CP, duration, slack
    w_cp = 1.2
    w_slack = 0.4
    w_scp = 0.4
    w_cone = 0.15
    w_res = 0.5
    w_dur = 0.3
    w_bn = 0.35

    score = (w_cp * norm_cp +
             w_slack * norm_slack_inv +
             w_scp * norm_scp +
             w_cone * norm_cone +
             w_res * (0.5 * norm_mr + 0.2 * norm_sr + 0.3 * norm_dw) +
             w_dur * norm_dur +
             w_bn * norm_bn)

    return score
# EVOLVE-BLOCK-END
