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
    
    # Pre-compute demand vectors for faster access if needed, but direct indexing is fine for n=30
    # demands is expected to be a numpy array of shape (n, n_res)
    
    while scheduled_count < n:
        # Find eligible: unscheduled with all predecessors done
        eligible = [i for i in range(n) if not scheduled[i] and remaining[i] == 0]
        if not eligible:
            break
        
        # Pick highest priority
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        best = eligible[0]
        
        # Find earliest feasible start considering predecessors
        es = 0.0
        for p in preds[best]:
            if sched_end[p] > es:
                es = sched_end[p]
        
        # Resource feasibility: scan forward
        start = es
        # Optimization: Only scan if there's a chance of conflict, but scanning is cheap for n=30
        # We scan from int(start)+1 to int(start)+4000. 
        # Note: start is float. If start is integer, we check t=start+1... 
        # The original code scanned from int(start)+1. 
        # If start is 5.0, int(5)+1 = 6. But activity can start at 5.0 if resources allow.
        # The check `sched_start[k] < t and sched_end[k] > t` defines resource usage at time t.
        # Usually, SGS checks if resources are available at the start time.
        # Let's stick to the logic that worked: scan integer times.
        
        # To be safe and consistent with previous successful runs, we scan integer time steps.
        # If start is 5.2, we check t=6, 7... 
        # Wait, if start is 5.2, the activity occupies [5.2, 5.2+dur].
        # Resources are typically checked at integer time points in discrete SGS or continuous.
        # The problem statement implies renewable resources. 
        # The previous code scanned `range(int(start) + 1, ...)`. 
        # If start=0, it checks 1, 2... It misses 0?
        # Let's re-verify the resource check logic.
        # `used` is calculated for time `t`.
        # If we start at `t`, we need resources for the duration.
        # The check `used + demands[best][j] > capacities[j]` at time `t` ensures that at time `t`, 
        # the total demand doesn't exceed capacity. 
        # If we start at 0, we need to check t=0.
        # The previous code started scanning from `int(start)+1`. 
        # If start=0, it checks 1, 2... It assumes t=0 is always free? Or that start is always integer?
        # In many SGS implementations, time is continuous, but resource checks are done at event points.
        # However, the previous code (score 1.042) used this scanning method and worked.
        # Let's look at the Inspiration code. It also used `range(int(start) + 1, ...)`.
        # This implies that if start=0, it checks t=1. 
        # This might be a bug if resources are tight at t=0.
        # However, `es` is calculated from predecessors. If no predecessors, es=0.
        # If we check t=1, and t=1 is free, we start at 1? No, we start at `start` which is `es`.
        # The loop sets `start = float(t)` if `t` is feasible.
        # So if `es=0` and `t=1` is the first feasible integer, it starts at 1.
        # This effectively pushes the start to the next integer if 0 is not checked.
        # Is 0 checked? No.
        # Let's modify the scan to start from `int(start)` to include the earliest possible integer.
        # But wait, if start=5.2, int(start)=5. We check 5. 
        # At t=5, is the activity active? Yes, [5.2, ...] includes 5.2, but not necessarily 5.0 if discrete.
        # If resources are discrete (integer time), an activity starting at 5.2 is not standard.
        # Usually, durations and starts are integers or we round.
        # Given the "30 activities" and standard RCPSP, integer times are likely.
        # Let's assume integer time steps for resource checks.
        # To be safe, let's scan from `int(start)` to `int(start) + 4000`.
        
        found = False
        # Check the exact earliest start first (continuous time)
        t0 = start
        ok0 = True
        for j in range(n_res):
            if demands[best][j] > 0:
                used = 0.0
                for k in range(n):
                    if scheduled[k] and sched_start[k] < t0 + durations[best] and (sched_end[k] > t0):
                        used += demands[k][j]
                if used + demands[best][j] > capacities[j] + 1e-9:
                    ok0 = False
                    break
        if ok0:
            found = True
        else:
            # Scan integer time points starting from ceil(t0)
            t_scan = int(math.ceil(t0))
            for t in range(t_scan, t_scan + 4001):
                ok = True
                for j in range(n_res):
                    if demands[best][j] > 0:
                        used = 0.0
                        for k in range(n):
                            if scheduled[k] and sched_start[k] < t + durations[best] and (sched_end[k] > t):
                                used += demands[k][j]
                        if used + demands[best][j] > capacities[j] + 1e-9:
                            ok = False
                            break
                if ok:
                    t0 = float(t)
                    found = True
                    break
        if not found:
            t0 = float(int(math.ceil(start)) + 4000)
        start = t0
            
        sched_start[best] = start
        sched_end[best] = start + durations[best]
        scheduled[best] = True
        scheduled_count += 1
        for v in successors[best]:
            remaining[v] -= 1
        if sched_end[best] > current:
            current = sched_end[best]
            
    return float(current)

def _compute_features(instance):
    """Compute a set of heuristic features for each activity."""
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

    # Hard CPL
    cpl = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # Smooth CPL (T=0.5)
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

    # Earliest start
    es = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            es[u] = 0.0
        else:
            es[u] = max(es[p] + durations[p] for p in preds[u])
    project_length = max(es[i] + durations[i] for i in range(n))

    # Slack
    slack = np.zeros(n, dtype=float)
    for i in range(n):
        slack[i] = (es[i] + durations[i] + cpl[i]) - project_length

    # Resource intensity (sum, max, prod)
    res_sum = np.zeros(n, dtype=float)
    res_max = np.zeros(n, dtype=float)
    res_prod = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        mx = 0.0
        pr = 1.0
        for j in range(n_res):
            r = demands[i][j] / capacities[j] if capacities[j] > 0 else 0.0
            s += r
            if r > mx:
                mx = r
            pr *= (1.0 + r)
        res_sum[i] = s
        res_max[i] = mx
        res_prod[i] = pr

    # Resource bottleneck priority: sum of ratio^1.5
    res_bottleneck = np.zeros(n, dtype=float)
    for i in range(n):
        bottleneck_score = 0.0
        for j in range(n_res):
            if demands[i][j] > 0:
                ratio = demands[i][j] / capacities[j] if capacities[j] > 0 else 0.0
                bottleneck_score += ratio ** 1.5
        res_bottleneck[i] = bottleneck_score

    # Resource blocking: for each activity, the fraction of project time it occupies
    # on its most constrained resource (demand/capacity * duration / project_length)
    project_len = max(es[i] + durations[i] for i in range(n)) if n > 0 else 1.0
    if project_len < 1e-9:
        project_len = 1.0
    res_blocking = np.zeros(n, dtype=float)
    for i in range(n):
        max_block = 0.0
        for j in range(n_res):
            if capacities[j] > 0 and demands[i][j] > 0:
                block = (demands[i][j] / capacities[j]) * (durations[i] / project_len)
                if block > max_block:
                    max_block = block
        res_blocking[i] = max_block

    # Predecessor count and depth
    pred_count = np.array([len(preds[i]) for i in range(n)], dtype=float)
    depth = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            depth[u] = 0.0
        else:
            depth[u] = max(depth[p] for p in preds[u]) + 1.0
    # Successor count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)

    # Resource-weighted tail: max of (res_max + tail of successors)
    res_tail = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_tail[u] = res_max[u]
        else:
            res_tail[u] = res_max[u] + max(res_tail[v] for v in successors[u])

    # Resource-weighted smooth tail (log-sum-exp)
    T2 = 0.3
    res_tail_smooth = np.zeros(n, dtype=float)
    for u in reverse_topo:
        if not successors[u]:
            res_tail_smooth[u] = res_max[u]
        else:
            sv = [res_tail_smooth[v] for v in successors[u]]
            m = max(sv)
            se = sum(math.exp((x - m) / T2) for x in sv)
            res_tail_smooth[u] = res_max[u] + m + T2 * math.log(se)

    # Normalize
    def norm(arr):
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    feats = {
        'cpl': norm(cpl),
        'scpl': norm(scpl),
        'slack_crit': 1.0 - norm(slack),
        'res_sum': norm(res_sum),
        'res_max': norm(res_max),
        'res_prod': norm(res_prod),
        'res_bottleneck': norm(res_bottleneck),
        'res_tail': norm(res_tail),
        'res_tail_smooth': norm(res_tail_smooth),
        'res_blocking': norm(res_blocking),
        'dur': norm(durations),
        'depth': norm(depth),
        'pred_count': norm(pred_count),
        'succ_count': norm(succ_count),
    }
    return feats, topo_order, successors

def priority(instance):
    start_time = time.time()
    DEADLINE = 145.0 # Safety margin from 150s limit
    feats, topo_order, successors = _compute_features(instance)
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])

    # Baseline weights (critical path and resource tails first)
    base = np.zeros(len(names))
    base[names.index('cpl')] = 1.0
    base[names.index('scpl')] = 0.3
    base[names.index('res_sum')] = 0.3
    base[names.index('res_bottleneck')] = 0.4
    base[names.index('res_tail')] = 0.5
    base[names.index('res_tail_smooth')] = 0.4
    base[names.index('res_blocking')] = 0.5
    base[names.index('slack_crit')] = 0.2
    base_weights = base
    base_scores = X @ base_weights
    best_scores = base_scores.copy()
    try:
        best_makespan = _schedule_makespan(durations, demands, capacities, preds, base_scores)
    except Exception:
        best_makespan = float('inf')

    # Coordinate descent on weights, fitting against makespan
    best_w = base_weights.copy()
    n_feat = len(names)
    step = 1.0
    improved_any = False
    while True:
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for k in range(n_feat):
            if time.time() - start_time > DEADLINE:
                break
            for delta in (step, -step, 0.5*step, -0.5*step, 0.25*step, -0.25*step):
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
                    improved_any = True
                    break
        if not improved:
            if step < 0.05:
                break
            step *= 0.5

    # Random restarts (always try some, not just when CD fails)
    rng = np.random.default_rng(0)
    attempts = 0
    max_attempts = 80
    while attempts < max_attempts and time.time() - start_time < DEADLINE:
        attempts += 1
        w = base_weights.copy()
        w += rng.normal(0, 0.3, size=n_feat)
        w = np.clip(w, -0.5, 3.0)
        scores = X @ w
        try:
            m = _schedule_makespan(durations, demands, capacities, preds, scores)
        except Exception:
            continue
        if m < best_makespan - 1e-9:
            best_makespan = m
            best_scores = scores.copy()
            best_w = w.copy()

    return best_scores
# EVOLVE-BLOCK-END
