# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _serial_sgs(durations, demands, capacities, predecessors, priority_vec):
    n = len(durations)
    if n == 0:
        return 0.0
    remaining_preds = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        pred_list = predecessors[j]
        if pred_list:
            remaining_preds[j] = len(pred_list)
            for p in pred_list:
                successors[p].append(j)
    heap = []
    for j in range(n):
        if remaining_preds[j] == 0:
            heapq.heappush(heap, (-float(priority_vec[j]), j))
    start = np.zeros(n)
    finish = np.zeros(n)
    res_free = np.zeros(len(capacities))
    scheduled = 0
    while heap:
        _, j = heapq.heappop(heap)
        ps = 0.0
        for p in predecessors[j]:
            if finish[p] > ps:
                ps = finish[p]
        rs = 0.0
        for k in range(len(capacities)):
            if demands[j][k] > 0 and res_free[k] > rs:
                rs = res_free[k]
        st = ps if ps > rs else rs
        start[j] = st
        finish[j] = st + durations[j]
        for k in range(len(capacities)):
            if demands[j][k] > 0 and finish[j] > res_free[k]:
                res_free[k] = finish[j]
        scheduled += 1
        for v in successors[j]:
            remaining_preds[v] -= 1
            if remaining_preds[v] == 0:
                heapq.heappush(heap, (-float(priority_vec[v]), v))
    if scheduled < n:
        return 1e9
    return float(np.max(finish))


def _min_cut_capacity(durations, demands, capacities, predecessors, successors):
    """
    Time-expanded max-flow per activity: for each activity j, compute the
    minimum resource capacity on any single resource that would be 'binding'
    if j were forced to be the bottleneck. Approximated by the maximum
    cumulative demand ratio over the critical window of j.
    Returns an array of required-capacity scores (higher = more constraining).
    """
    n = len(durations)
    n_res = len(capacities)
    if n == 0 or n_res == 0:
        return np.zeros(n)

    # Longest path from start (earliest start) and to end (critical tail)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        in_deg[j] = len(predecessors[j]) if predecessors[j] else 0

    # Topological order
    from collections import deque
    q = deque([j for j in range(n) if in_deg[j] == 0])
    topo = []
    indeg2 = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    if len(topo) < n:
        for j in range(n):
            if j not in topo:
                topo.append(j)

    es = np.zeros(n)  # earliest start
    for u in topo:
        for v in successors[u]:
            cand = es[u] + durations[u]
            if cand > es[v]:
                es[v] = cand

    # Latest finish / latest start (backward pass)
    lf = np.zeros(n)
    # project makespan upper bound
    ms_ub = np.max(es + durations) if n > 0 else 0.0
    lf[:] = ms_ub
    for u in reversed(topo):
        for v in successors[u]:
            cand = lf[v] - durations[v]
            if cand < lf[u]:
                lf[u] = cand
    ls = lf - durations

    # For each activity, the "critical window" is [es[j], lf[j]].
    # Required capacity: for each resource k, the max over time in window of
    # (cumulative demand of activities that could overlap j on k) / capacity_k.
    # Approximate: peak concurrent demand on each resource within the window,
    # then take the max ratio across resources.
    req_cap = np.zeros(n)
    for j in range(n):
        lo = es[j]
        hi = lf[j]
        # Activities that overlap j's window on time AND share a resource
        peak_ratio = 0.0
        for k in range(n_res):
            if capacities[k] <= 0:
                continue
            # sum demands of activities whose [es, es+dur] overlaps [lo, hi]
            total = 0.0
            for i in range(n):
                if demands[i][k] > 0:
                    i_start = es[i]
                    i_end = es[i] + durations[i]
                    if i_end > lo and i_start < hi:
                        total += demands[i][k]
            ratio = total / capacities[k]
            if ratio > peak_ratio:
                peak_ratio = ratio
        req_cap[j] = peak_ratio
    return req_cap


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)

    # Longest path to end (critical tail)
    in_deg = np.zeros(n, dtype=int)
    for j in range(n):
        in_deg[j] = len(predecessors[j]) if predecessors[j] else 0
    from collections import deque
    q = deque([j for j in range(n) if in_deg[j] == 0])
    topo = []
    indeg2 = in_deg.copy()
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg2[v] -= 1
            if indeg2[v] == 0:
                q.append(v)
    if len(topo) < n:
        for j in range(n):
            if j not in topo:
                topo.append(j)

    rev_topo = list(reversed(topo))
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_topo:
        max_succ = 0.0
        for j in successors[i]:
            cand = durations[j] + longest_path_to_end[j]
            if cand > max_succ:
                max_succ = cand
        longest_path_to_end[i] = durations[i] + max_succ

    # Resource pressure components
    n_res = len(capacities)
    max_ratio = np.zeros(n)
    sum_ratio = np.zeros(n)
    for j in range(n):
        mx = 0.0
        s = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
        max_ratio[j] = mx
        sum_ratio[j] = s

    succ_count = np.zeros(n)
    for j in range(n):
        succ_count[j] = len(successors[j])

    def _norm(x):
        m = np.max(x)
        if m > 0:
            return x / m
        return np.zeros_like(x)

    norm_cp = _norm(longest_path_to_end)
    norm_max_ratio = _norm(max_ratio)
    norm_sum_ratio = _norm(sum_ratio)
    norm_succ = _norm(succ_count)

    def _base_score():
        return norm_cp * (1.0 + 0.6 * norm_max_ratio + 0.4 * norm_sum_ratio)

    if _remaining_budget() < 30.0:
        return _base_score()

    # Compute bottleneck required-capacity score (inverse used as priority boost)
    req_cap = _min_cut_capacity(durations, demands, capacities, predecessors, successors)
    norm_req = _norm(req_cap)
    # Inverse: activities requiring MORE capacity to be bottleneck get HIGHER priority
    # (they are the true bottlenecks). Use 1/(1+req) normalized, or just norm_req.
    # We'll blend: priority = norm_cp * (1 + a*norm_max_ratio + b*norm_sum_ratio + c*norm_req)

    best_score = _base_score()
    best_ms = _serial_sgs(durations, demands, capacities, predecessors, best_score)

    # Grid search over (a, b, c) where c is the bottleneck-capacity weight
    a_vals = [0.0, 0.4, 0.6, 0.8, 1.2]
    b_vals = [0.0, 0.2, 0.4, 0.6]
    c_vals = [0.0, 0.2, 0.4, 0.6, 0.8]

    base = norm_cp
    evaluated = 0
    max_evals = 200
    for a in a_vals:
        if _remaining_budget() < 30.0:
            break
        for b in b_vals:
            if _remaining_budget() < 30.0:
                break
            for c in c_vals:
                if _remaining_budget() < 30.0 or evaluated >= max_evals:
                    break
                cand = base * (1.0 + a * norm_max_ratio + b * norm_sum_ratio + c * norm_req)
                ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                evaluated += 1
                if ms < best_ms:
                    best_ms = ms
                    best_score = cand

    # Finer polish around best if budget remains
    if _remaining_budget() > 45.0:
        # Try a few refined combinations
        for a2 in [0.5, 0.7, 0.9, 1.0]:
            if _remaining_budget() < 30.0:
                break
            for b2 in [0.3, 0.5, 0.7]:
                if _remaining_budget() < 30.0:
                    break
                for c2 in [0.3, 0.5, 0.7, 1.0]:
                    if _remaining_budget() < 30.0:
                        break
                    cand = base * (1.0 + a2 * norm_max_ratio + b2 * norm_sum_ratio + c2 * norm_req)
                    ms = _serial_sgs(durations, demands, capacities, predecessors, cand)
                    if ms < best_ms:
                        best_ms = ms
                        best_score = cand

    return best_score
# EVOLVE-BLOCK-END
