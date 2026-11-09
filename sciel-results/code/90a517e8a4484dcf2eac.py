import numpy as np
import time
from collections import deque

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before the 150s kill


def _build_graph(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    if n == 0:
        return durations, demands, capacities, predecessors, [], [], np.zeros(0, int)
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    if len(topo) != n:
        return durations, demands, capacities, predecessors, [], [], in_degree
    return durations, demands, capacities, predecessors, topo, successors, in_degree


def _cpm(durations, topo, successors, n):
    """Vectorized critical path: earliest/latest start, total float, criticality."""
    es = np.zeros(n, dtype=float)
    for u in topo:
        preds = [v for v in range(n) if u in _preds_cache[v]]  # placeholder, replaced below
    # We need predecessors; recompute es via successors of predecessors is awkward,
    # so we pass a pred list instead. (Handled in _features.)
    raise RuntimeError("use _features")


def _features(instance, topo, successors):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    n_res = len(capacities) if capacities.ndim > 0 else 0

    # predecessors as list of lists for fast access
    pred = [list(predecessors[i]) for i in range(n)]

    # ---- CPM forward pass (earliest start) ----
    es = np.zeros(n, dtype=float)
    for u in topo:
        if pred[u]:
            es[u] = max(es[p] + durations[p] for p in pred[u])
    project_end = float(np.max(es + durations)) if n else 0.0

    # ---- CPM backward pass (latest start) ----
    ls = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if not successors[u]:
            ls[u] = project_end - durations[u]
        else:
            ls[u] = min(ls[v] - durations[v] for v in successors[u])

    # ---- Critical Path Length (tail) from each node to project end ----
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo):
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # ---- total float ----
    total_float = np.clip(ls - es, 0.0, None)

    # ---- criticality: fraction of longest tail that is "tight" ----
    # An activity is critical if it lies on some longest path.
    criticality = np.zeros(n, dtype=float)
    for u in topo:
        # is there a path from start to u with zero float AND from u to end with zero float?
        # approximate: forward longest path to u
        pass
    # forward longest path to each node
    fwd = np.zeros(n, dtype=float)
    for u in topo:
        if pred[u]:
            fwd[u] = max(fwd[p] + durations[p] for p in pred[u])
    # criticality: node is on a longest path if fwd[u] + cpl[u] == project_end (within eps)
    on_crit = (fwd + cpl - project_end) <= 1e-6
    criticality = on_crit.astype(float)

    # ---- resource pressure signals (vectorized) ----
    sum_ratio = np.zeros(n, dtype=float)
    max_ratio = np.zeros(n, dtype=float)
    soft_max_ratio = np.zeros(n, dtype=float)
    soft_min_ratio = np.zeros(n, dtype=float)
    if n_res > 0:
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        ratios = demands / safe_cap  # (n, n_res)
        mask = (capacities[None, :] > 0) & (demands > 0)  # (n, n_res)
        masked = np.where(mask, ratios, 0.0)
        sum_ratio = masked.sum(axis=1)
        max_ratio = masked.max(axis=1)
        # soft-max (smooth max) with temperature
        temp = 0.15
        m = masked.max(axis=1, keepdims=True)
        exp_terms = np.exp((masked - m) / temp)
        soft_max_ratio = (m + temp * np.log(exp_terms.sum(axis=1, keepdims=True))).ravel()
        # soft-min (smooth min) of positive ratios: tightest resource constraint
        pos = np.where(mask, ratios, np.inf)
        mmin = pos.min(axis=1, keepdims=True)
        exp_min = np.exp(-(pos - mmin) / temp)
        exp_min = np.where(np.isfinite(pos), exp_min, 0.0)
        soft_min_ratio = (mmin - temp * np.log(exp_min.sum(axis=1, keepdims=True))).ravel()
        soft_min_ratio = np.where(np.isfinite(soft_min_ratio), soft_min_ratio, 0.0)

    # ---- resource-weighted tail (smooth): longest path where each successor's
    # duration is boosted by a smooth function of its resource pressure ----
    resource_tail = np.zeros(n, dtype=float)
    if n_res > 0:
        w = 1.0 + 0.3 * sum_ratio
        for u in reversed(topo):
            if not successors[u]:
                resource_tail[u] = durations[u]
            else:
                cands = durations[successors[u]] * w[successors[u]] + resource_tail[successors[u]]
                resource_tail[u] = durations[u] + float(np.max(cands))

    # ---- demand-profile bottleneck via sweep line (vectorized per resource) ----
    # For each resource, build the demand profile over time at earliest starts and
    # measure each activity's contribution to the peak (how "bottleneck" it is).
    bottleneck = np.zeros(n, dtype=float)
    if n_res > 0 and n > 0:
        T = int(np.ceil(project_end)) + 2
        for k in range(n_res):
            if capacities[k] <= 0:
                continue
            d = demands[:, k]
            if not np.any(d > 0):
                continue
            start = np.floor(es).astype(int)
            end = np.ceil(es + durations).astype(int)
            end = np.clip(end, 0, T)
            start = np.clip(start, 0, T)
            # sweep line: events at start (+d) and end (-d)
            profile = np.zeros(T + 1, dtype=float)
            # accumulate using np.add.at for correctness with repeated times
            np.add.at(profile, start, d)
            np.add.at(profile, end, -d)
            profile = np.cumsum(profile)
            peak = float(np.max(profile))
            if peak <= 1e-10:
                continue
            # activity's bottleneck contribution = demand/peak for active interval,
            # approximated by demand/peak (activities active at the peak matter most)
            # weight by how much of the peak this activity occupies at its own active window
            active = d > 0
            bottleneck[active] += d[active] / peak

    # ---- normalize ----
    def safe_norm(arr):
        m = float(np.max(arr)) if arr.size else 0.0
        if m <= 1e-10:
            return np.zeros_like(arr, dtype=float)
        return arr / m

    return {
        "n": n, "n_res": n_res,
        "durations": durations, "demands": demands, "capacities": capacities,
        "pred": pred, "successors": successors,
        "norm_cpl": safe_norm(cpl),
        "norm_rt": safe_norm(resource_tail),
        "norm_sum": safe_norm(sum_ratio),
        "norm_max": safe_norm(max_ratio),
        "norm_soft": safe_norm(soft_max_ratio),
        "norm_softmin": safe_norm(soft_min_ratio),
        "norm_dur": safe_norm(durations),
        "norm_float": safe_norm(total_float),
        "crit": criticality,
        "norm_bottleneck": safe_norm(bottleneck),
    }


def _serial_sgs(priority, durations, demands, capacities, pred, successors, n):
    """Serial schedule-generation scheme. Returns makespan (float)."""
    n_res = len(capacities) if capacities.ndim > 0 else 0
    remaining_cap = np.asarray(capacities, dtype=float).copy()
    remaining_pred = [len(pred[i]) for i in range(n)]
    done = [False] * n
    time = 0.0
    makespan = 0.0
    # event queue of (finish_time, activity)
    events = []  # simple list, we pop the min
    eligible = []  # list of eligible activity ids

    for i in range(n):
        if remaining_pred[i] == 0:
            eligible.append(i)

    while True:
        # start highest-priority eligible activity that fits
        started = False
        # sort eligible by priority desc
        if eligible:
            eligible.sort(key=lambda x: -priority[x])
            for i in eligible:
                if done[i]:
                    continue
                # check resource fit
                ok = True
                for k in range(n_res):
                    if demands[i, k] > remaining_cap[k] + 1e-9:
                        ok = False
                        break
                if ok:
                    # start now
                    for k in range(n_res):
                        remaining_cap[k] -= demands[i, k]
                    done[i] = True
                    ft = time + durations[i]
                    events.append((ft, i))
                    makespan = max(makespan, ft)
                    # release successors
                    for v in successors[i]:
                        remaining_pred[v] -= 1
                        if remaining_pred[v] == 0:
                            eligible.append(v)
                    started = True
                    break
        if not started:
            # no eligible activity can start; advance time to next finish
            if not events:
                break
            # pop earliest finish
            events.sort(key=lambda x: x[0])
            ft, i = events.pop(0)
            time = ft
            for k in range(n_res):
                remaining_cap[k] += demands[i, k]
        # loop

    return makespan


def _grid_scores(feats):
    """Generate candidate weight vectors (a small grid)."""
    n = feats["n"]
    base = {
        "cpl": feats["norm_cpl"],
        "rt": feats["norm_rt"],
        "soft": feats["norm_soft"],
        "softmin": feats["norm_softmin"],
        "sum": feats["norm_sum"],
        "max": feats["norm_max"],
        "bottleneck": feats["norm_bottleneck"],
        "dur": feats["norm_dur"],
        "float": feats["norm_float"],
    }
    # candidate (alpha, beta, gamma, delta, eps) tuples
    # score = (alpha*cpl + beta*rt) * (1 + gamma*(soft*0.5+softmin*0.5)) + delta*bottleneck + eps*dur
    alphas = [0.6, 0.7, 0.8]
    betas = [0.2, 0.3, 0.4]
    gammas = [0.3, 0.5, 0.8]
    deltas = [0.0, 0.1, 0.2]
    epsilons = [0.0, 0.05]
    cands = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                for d in deltas:
                    for e in epsilons:
                        cands.append((a, b, g, d, e))
    return cands, base


def priority(instance):
    if time.time() - _START_TIME > _DEADLINE:
        return np.asarray(instance["durations"], dtype=float)

    durations, demands, capacities, predecessors, topo, successors, in_degree = _build_graph(instance)
    n = len(durations)
    if n == 0:
        return np.array([])
    if len(topo) != n:
        return durations.copy()

    feats = _features(instance, topo, successors)

    # Build the feature matrix rows once
    base_feats = feats
    # Precompute the "structural" and "resource" parts for fast scoring
    # score = (a*cpl + b*rt) * (1 + g*(0.5*soft + 0.5*softmin)) + d*bottleneck + e*dur
    cpl = base_feats["norm_cpl"]
    rt = base_feats["norm_rt"]
    soft = 0.5 * base_feats["norm_soft"] + 0.5 * base_feats["norm_softmin"]
    bottleneck = base_feats["norm_bottleneck"]
    dur = base_feats["norm_dur"]

    # Candidate weight grid
    alphas = [0.6, 0.7, 0.8]
    betas = [0.2, 0.3, 0.4]
    gammas = [0.3, 0.5, 0.8]
    deltas = [0.0, 0.1, 0.2]
    epsilons = [0.0, 0.05]

    pred = base_feats["pred"]
    succ = base_feats["successors"]
    n_res = base_feats["n_res"]

    best_score_vec = None
    best_makespan = float("inf")

    # If too little time, skip search and use a sensible default
    if time.time() - _START_TIME > _DEADLINE - 20:
        a, b, g, d, e = 0.7, 0.3, 0.5, 0.1, 0.05
        score = (a * cpl + b * rt) * (1.0 + g * soft) + d * bottleneck + e * dur
        return score

    # Self-calibrating search over the weight grid
    for a in alphas:
        for b in betas:
            for g in gammas:
                for d in deltas:
                    for e in epsilons:
                        if time.time() - _START_TIME > _DEADLINE - 10:
                            break
                        score = (a * cpl + b * rt) * (1.0 + g * soft) + d * bottleneck + e * dur
                        mk = _serial_sgs(score, durations, demands, capacities, pred, succ, n)
                        if mk < best_makespan:
                            best_makespan = mk
                            best_score_vec = score.copy()
                    if time.time() - _START_TIME > _DEADLINE - 10:
                        break
                if time.time() - _START_TIME > _DEADLINE - 10:
                    break
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
        if time.time() - _START_TIME > _DEADLINE - 10:
            break

    if best_score_vec is None:
        # fallback default
        a, b, g, d, e = 0.7, 0.3, 0.5, 0.1, 0.05
        best_score_vec = (a * cpl + b * rt) * (1.0 + g * soft) + d * bottleneck + e * dur

    return best_score_vec
