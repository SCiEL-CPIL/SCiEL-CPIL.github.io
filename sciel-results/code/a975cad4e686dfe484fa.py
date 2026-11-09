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

    # 1) Smooth critical path: log-sum-exp over successor chains
    #    This captures near-critical paths, not just the single longest.
    #    cp[i] = durations[i] + soft_max_j(durations[j] + cp[j])
    #    where soft_max uses log-sum-exp with temperature parameter.
    TEMP = 2.0  # lower = more peaked (closer to hard max), higher = smoother
    cp = np.zeros(n, dtype=float)
    for i in rev_top_order:
        succ_vals = []
        for j in successors[i]:
            succ_vals.append(durations[j] + cp[j])
        if succ_vals:
            arr = np.array(succ_vals, dtype=float)
            m = np.max(arr)
            # log-sum-exp: m + log(sum(exp(arr - m)))
            lse = m + np.log(np.sum(np.exp(arr - m)))
            # Smooth max: TEMP * log(sum(exp(arr/TEMP))) = TEMP * (m/TEMP + log(sum(exp((arr-m)/TEMP))))
            smooth_max = TEMP * (m / TEMP + np.log(np.sum(np.exp((arr - m) / TEMP))))
        else:
            smooth_max = 0.0
        cp[i] = durations[i] + smooth_max

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

    # 3) Resource pressure: duration-weighted bottleneck and sum
    #    Weighting by duration emphasizes activities that occupy resources longer.
    n_res = len(capacities)
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
    dur_weighted_max = np.zeros(n, dtype=float)
    dur_weighted_sum = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        s = 0.0
        dw_mx = 0.0
        dw_s = 0.0
        for k in range(n_res):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
                s += r
                # duration-weighted: how much total resource-time this activity consumes relative to capacity
                dw_r = durations[j] * r
                if dw_r > dw_mx:
                    dw_mx = dw_r
                dw_s += dw_r
        max_ratio[j] = mx
        sum_ratio[j] = s
        dur_weighted_max[j] = dw_mx
        dur_weighted_sum[j] = dw_s

    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr

    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    max_dwm = float(np.max(dur_weighted_max)) if n > 0 else 0.0
    if max_dwm == 0:
        max_dwm = 1.0
    norm_dwm = dur_weighted_max / max_dwm

    max_dws = float(np.max(dur_weighted_sum)) if n > 0 else 0.0
    if max_dws == 0:
        max_dws = 1.0
    norm_dws = dur_weighted_sum / max_dws

    # Normalize critical path
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine. Primary: smooth critical path.
    # Secondary: transitive unlock (front-load gatekeepers).
    # Tertiary: duration-weighted resource pressure (captures both how
    # resource-intensive and how long the activity is).
    w_cp = 1.0
    w_cone = 0.30
    w_res = 0.15
    # Blend raw ratio and duration-weighted for resource signal
    res_signal = 0.4 * norm_mr + 0.2 * norm_sr + 0.3 * norm_dwm + 0.1 * norm_dws
    score = w_cp * norm_cp + w_cone * norm_cone + w_res * res_signal

    return score
# EVOLVE-BLOCK-END
