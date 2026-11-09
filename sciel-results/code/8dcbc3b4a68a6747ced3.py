# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque
import math

def _schedule_makespan(durations, demands, capacities, preds, priority_scores):
    """Serial schedule generation scheme. Returns the makespan."""
    n = len(durations)
    n_res = len(capacities)
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1
    remaining = in_degree.copy()
    sched_start = np.zeros(n)
    sched_end = np.zeros(n)
    scheduled = [False] * n
    current = 0.0
    scheduled_count = 0
    while scheduled_count < n:
        eligible = [i for i in range(n) if not scheduled[i] and remaining[i] == 0]
        if not eligible:
            break
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        best = eligible[0]
        es = 0.0
        for p in preds[best]:
            if sched_end[p] > es:
                es = sched_end[p]
        start = es
        for t in range(int(start) + 1, int(start) + 4001):
            ok = True
            for j in range(n_res):
                if demands[best][j] > 0:
                    used = 0.0
                    for k in range(n):
                        if scheduled[k] and sched_start[k] < t and sched_end[k] > t:
                            used += demands[k][j]
                    if used + demands[best][j] > capacities[j] + 1e-9:
                        ok = False
                        break
            if ok:
                start = float(t)
                break
        else:
            start = float(int(start) + 4000)
        sched_start[best] = start
        sched_end[best] = start + durations[best]
        scheduled[best] = True
        scheduled_count += 1
        for v in successors[best]:
            remaining[v] -= 1
        if sched_end[best] > current:
            current = sched_end[best]
    return float(current)

def _time_expanded_capacity(durations, demands, capacities, preds, successors, topo_order, reverse_topo):
    """
    For each activity i, compute the minimum resource capacity that would be required
    if i were forced onto the critical path (earliest start, latest finish).
    We approximate this by:
      1. Compute earliest start (ES) and latest start (LS) under precedence only.
      2. For each activity, simulate a schedule where it starts at ES (or LS if tighter),
         and other activities are scheduled greedily. Track peak resource usage.
      3. The "required capacity" for activity i is the peak usage that i contributes to.
    We use a lighter approach: for each resource j, compute the peak concurrent demand
    in a baseline schedule, then attribute how much of that peak is due to each activity.
    """
    n = len(durations)
    n_res = len(capacities)
    
    # Earliest start / earliest finish
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(ef[p] for p in preds[u])
        ef[u] = es[u] + durations[u]
    
    # Latest start / latest finish (backward pass)
    project_end = max(ef)
    lf = np.zeros(n, dtype=float)
    ls = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            lf[u] = project_end
        else:
            lf[u] = min(ls[v] for v in successors[u])
        ls[u] = lf[u] - durations[u]
    
    # Slack
    slack = ls - es
    
    # Build a baseline serial schedule using CPL priority to get actual start times
    cpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])
    
    # Do a quick serial schedule to get actual start times
    in_deg = np.zeros(n, dtype=int)
    for i in range(n):
        in_deg[i] = len(preds[i])
    rem = in_deg.copy()
    sched_start = np.zeros(n)
    sched_end = np.zeros(n)
    done = [False] * n
    cnt = 0
    while cnt < n:
        elig = [i for i in range(n) if not done[i] and rem[i] == 0]
        if not elig:
            break
        elig.sort(key=lambda i: cpl[i], reverse=True)
        b = elig[0]
        s = 0.0
        for p in preds[b]:
            if sched_end[p] > s:
                s = sched_end[p]
        # Resource check
        for t in range(int(s) + 1, int(s) + 4001):
            ok = True
            for j in range(n_res):
                if demands[b][j] > 0:
                    used = 0.0
                    for k in range(n):
                        if done[k] and sched_start[k] < t and sched_end[k] > t:
                            used += demands[k][j]
                    if used + demands[b][j] > capacities[j] + 1e-9:
                        ok = False
                        break
            if ok:
                s = float(t)
                break
        else:
            s = float(int(s) + 4000)
        sched_start[b] = s
        sched_end[b] = s + durations[b]
        done[b] = True
        cnt += 1
        for v in successors[b]:
            rem[v] -= 1
    
    # For each activity, compute its "bottleneck contribution":
    # At each time point where this activity is active, what is the total resource usage?
    # The peak concurrent usage while this activity is running indicates how much
    # capacity is "required" due to this activity's presence.
    # We compute: for activity i, for each resource j, the max over time t in [start_i, end_i)
    # of (sum of demands[k][j] for all k active at t). Then the "required capacity" is
    # the max over j of that peak. The priority is inverse: lower required capacity = less constraining.
    # But we want to prioritize activities that MOST constrain, so higher score = more constraining.
    
    # Efficient: collect all unique event times
    event_times = set()
    for i in range(n):
        event_times.add(int(sched_start[i]))
        event_times.add(int(sched_end[i]))
    event_times = sorted(event_times)
    
    # For each event time, compute resource usage
    # For each activity, track the peak resource usage during its active interval
    peak_usage = np.zeros((n, n_res), dtype=float)
    
    for i in range(n):
        s_i = int(sched_start[i])
        e_i = int(sched_end[i])
        for j in range(n_res):
            if demands[i][j] == 0:
                continue
            peak = 0.0
            for t in event_times:
                if t >= s_i and t < e_i:
                    used = 0.0
                    for k in range(n):
                        if int(sched_start[k]) <= t < int(sched_end[k]):
                            used += demands[k][j]
                    if used > peak:
                        peak = used
            peak_usage[i, j] = peak
    
    # Required capacity for activity i = max over resources of peak usage during i's interval
    # But we want the MARGINAL contribution: how much does i add to the peak?
    # Approximate: required_cap[i] = max_j peak_usage[i,j]
    # Priority: we want high priority for activities that are on the critical bottleneck.
    # Use: score[i] = max_j (peak_usage[i,j] / capacities[j]) if capacities[j] > 0
    
    bottleneck_score = np.zeros(n, dtype=float)
    for i in range(n):
        mx = 0.0
        for j in range(n_res):
            if capacities[j] > 0 and demands[i][j] > 0:
                ratio = peak_usage[i, j] / capacities[j]
                if ratio > mx:
                    mx = ratio
        bottleneck_score[i] = mx
    
    return bottleneck_score, es, ef, ls, lf, slack, cpl

def _compute_features(instance):
    """Compute features including time-expanded capacity bottleneck."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n_res = len(capacities)

    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    topo_order = []
    rem = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            rem[v] -= 1
            if rem[v] == 0:
                queue.append(v)
    if len(topo_order) != n:
        topo_order = list(range(n))
    reverse_topo = list(reversed(topo_order))

    # Time-expanded capacity bottleneck (new structural feature)
    te_bottleneck, es, ef, ls, lf, slack, cpl = _time_expanded_capacity(
        durations, demands, capacities, preds, successors, topo_order, reverse_topo
    )

    # Smooth CPL
    T = 0.5
    scpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            scpl[u] = durations[u]
        else:
            sv = [scpl[v] for v in successors[u]]
            m = max(sv)
            se = sum(math.exp((x - m) / T) for x in sv)
            scpl[u] = durations[u] + m + T * math.log(se)

    project_length = max(ef)
    slack_norm = np.zeros(n, dtype=float)
    for i in range(n):
        slack_norm[i] = (es[i] + durations[i] + cpl[i]) - project_length

    # Resource intensity
    res_sum = np.zeros(n, dtype=float)
    res_max = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        mx = 0.0
        for j in range(n_res):
            r = demands[i][j] / capacities[j] if capacities[j] > 0 else 0.0
            s += r
            if r > mx:
                mx = r
        res_sum[i] = s
        res_max[i] = mx

    pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    depth = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            depth[u] = 0.0
        else:
            depth[u] = max(depth[p] for p in preds[u]) + 1.0
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    res_tail = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_tail[u] = res_max[u]
        else:
            res_tail[u] = res_max[u] + max(res_tail[v] for v in successors[u])

    def norm(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    feats = {
        'cpl': norm(cpl),
        'scpl': norm(scpl),
        'te_bottleneck': norm(te_bottleneck),
        'slack_crit': 1.0 - norm(slack_norm),
        'res_sum': norm(res_sum),
        'res_max': norm(res_max),
        'res_tail': norm(res_tail),
        'dur': norm(durations),
        'depth': norm(depth),
        'pred_count': norm(pred_count),
        'succ_count': norm(succ_count),
    }
    return feats, topo_order, successors

def priority(instance):
    start_time = time.time()
    DEADLINE = 140.0
    feats, topo_order, successors = _compute_features(instance)
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])

    # Baseline: emphasize time-expanded bottleneck + CPL
    base = np.zeros(len(names))
    base[names.index('cpl')] = 1.0
    base[names.index('te_bottleneck')] = 1.5
    base[names.index('scpl')] = 0.3
    base[names.index('res_tail')] = 0.5
    base[names.index('slack_crit')] = 0.2
    base_weights = base
    base_scores = X @ base_weights
    best_scores = base_scores.copy()
    try:
        best_makespan = _schedule_makespan(durations, demands, capacities, preds, base_scores)
    except Exception:
        best_makespan = float('inf')

    # Coordinate descent
    best_w = base_weights.copy()
    n_feat = len(names)
    step = 0.5
    while True:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for k in range(n_feat):
            if time.time() - start_time > DEADLINE:
                break
            for delta in (step, -step, 0.5*step, -0.5*step):
                w = best_w.copy()
                w[k] += delta
                if (w < -0.5).any() or (w > 3.0).any():
                    continue
                scores = X @ w
                try:
                    m = _schedule_makespan(durations, demands, capacities, preds, scores)
                except Exception:
                    continue
                if m < best_makespan - 1e-9:
                    best_makespan = m
                    best_w = w.copy()
                    best_scores = scores.copy()
                    improved = True
                    break
        if not improved:
            if step < 0.05:
                break
            step *= 0.7

    # Random restarts
    rng = np.random.default_rng(0)
    attempts = 0
    while attempts < 60 and time.time() - start_time < DEADLINE - 5.0:
        attempts += 1
        w = base_weights.copy()
        w += rng.normal(0, 0.4, size=n_feat)
        w = np.clip(w, -0.5, 3.0)
        scores = X @ w
        try:
            m = _schedule_makespan(durations, demands, capacities, preds, scores)
        except Exception:
            continue
        if m < best_makespan - 1e-9:
            best_makespan = m
            best_w = w.copy()
            best_scores = scores.copy()

    # Polish
    if time.time() - start_time < DEADLINE - 3.0:
        rng = np.random.default_rng(1)
        for _ in range(30):
            if time.time() - start_time > DEADLINE:
                break
            w = best_w + rng.normal(0, 0.05, size=n_feat)
            w = np.clip(w, -0.5, 3.0)
            scores = X @ w
            try:
                m = _schedule_makespan(durations, demands, capacities, preds, scores)
            except Exception:
                continue
            if m < best_makespan - 1e-9:
                best_makespan = m
                best_w = w.copy()
                best_scores = scores.copy()

    return best_scores
# EVOLVE-BLOCK-END
