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

    # 3) Resource pressure via time-expanded max-flow bottleneck analysis
    #    For each activity, estimate how much it constrains the resource profile
    #    by computing a flow-based "required capacity" on a simplified time-expanded
    #    network. Activities that force higher required capacity get higher priority.
    n_res = len(capacities)
    
    # First compute a quick serial schedule (ES/EF) using the current CP as tie-breaker
    # to get a rough time horizon for the time-expanded network
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    for i in top_order:
        for p in predecessors[i]:
            if ef[p] > es[i]:
                es[i] = ef[p]
        ef[i] = es[i] + durations[i]
    
    T_max = float(np.max(ef)) if n > 0 else 1.0
    if T_max <= 0:
        T_max = 1.0
    
    # For each resource, compute the "peak demand profile" under the serial schedule
    # and identify which activities contribute most to the peak.
    # Use a max-flow formulation: for each resource k, build a time-expanded network
    # where the max flow from source to sink equals the minimum capacity needed.
    # The contribution of each activity to the bottleneck is its share of the peak.
    
    # Simplified approach: compute per-resource peak overlap
    # For activity j on resource k, its "bottleneck contribution" is proportional
    # to how much it participates in the peak resource usage.
    
    bottleneck_score = np.zeros(n, dtype=float)
    dur_weighted_bottleneck = np.zeros(n, dtype=float)
    
    for k in range(n_res):
        if capacities[k] <= 0:
            continue
        cap_k = capacities[k]
        
        # Compute resource usage profile over time (discretized by activity boundaries)
        # Time points are all ES and EF values
        time_points = set()
        for j in range(n):
            time_points.add(es[j])
            time_points.add(ef[j])
        time_points = sorted(time_points)
        
        if len(time_points) < 2:
            continue
        
        # Compute peak demand for this resource
        peak_demand = 0.0
        peak_time = 0.0
        for idx in range(len(time_points) - 1):
            t0 = time_points[idx]
            t1 = time_points[idx + 1]
            if t1 <= t0:
                continue
            demand_at_t = 0.0
            for j in range(n):
                if demands[j][k] > 0 and es[j] < t1 and ef[j] > t0:
                    demand_at_t += demands[j][k]
            if demand_at_t > peak_demand:
                peak_demand = demand_at_t
                peak_time = (t0 + t1) / 2.0
        
        if peak_demand <= 0:
            continue
        
        # For each activity, compute its "required capacity contribution"
        # An activity contributes to the bottleneck if it overlaps with the peak
        # and its demand is a significant fraction of the peak.
        # Use a smooth surrogate: the log-sum-exp of (demand_j / peak) for activities
        # that overlap the peak, weighted by duration overlap with peak window.
        
        # Find activities overlapping the peak window
        peak_window = 0.0
        # Estimate peak window width
        for idx in range(len(time_points) - 1):
            t0 = time_points[idx]
            t1 = time_points[idx + 1]
            if t1 <= t0:
                continue
            demand_at_t = 0.0
            for j in range(n):
                if demands[j][k] > 0 and es[j] < t1 and ef[j] > t0:
                    demand_at_t += demands[j][k]
            if abs(demand_at_t - peak_demand) < 1e-9:
                peak_window = max(peak_window, t1 - t0)
        
        if peak_window <= 0:
            peak_window = T_max / n if n > 0 else 1.0
        
        # For each activity, compute overlap with peak window and demand ratio
        for j in range(n):
            if demands[j][k] <= 0:
                continue
            # Overlap between [es[j], ef[j]] and peak vicinity
            # Use a Gaussian-like weight centered at peak_time
            center_j = (es[j] + ef[j]) / 2.0
            dist = abs(center_j - peak_time)
            # Smooth weight: exp(-dist^2 / (2 * sigma^2))
            sigma = peak_window if peak_window > 0 else 1.0
            weight = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
            
            # Demand ratio relative to peak
            ratio = demands[j][k] / peak_demand
            
            # Contribution to bottleneck
            bottleneck_score[j] += weight * ratio
            dur_weighted_bottleneck[j] += weight * ratio * durations[j]
    
    # Normalize
    max_bs = float(np.max(bottleneck_score)) if n > 0 else 0.0
    if max_bs == 0:
        max_bs = 1.0
    norm_bs = bottleneck_score / max_bs
    
    max_dwb = float(np.max(dur_weighted_bottleneck)) if n > 0 else 0.0
    if max_dwb == 0:
        max_dwb = 1.0
    norm_dwb = dur_weighted_bottleneck / max_dwb
    
    # Also keep simple ratio signals as fallback
    max_ratio = np.zeros(n, dtype=float)
    sum_ratio = np.zeros(n, dtype=float)
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
    
    max_mr = float(np.max(max_ratio)) if n > 0 else 0.0
    if max_mr == 0:
        max_mr = 1.0
    norm_mr = max_ratio / max_mr
    
    max_sr = float(np.max(sum_ratio)) if n > 0 else 0.0
    if max_sr == 0:
        max_sr = 1.0
    norm_sr = sum_ratio / max_sr

    # Normalize critical path
    max_cp = float(np.max(cp)) if n > 0 else 0.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = cp / max_cp

    # Combine. Primary: smooth critical path.
    # Secondary: transitive unlock (front-load gatekeepers).
    # Tertiary: flow-based bottleneck (captures which activities most constrain
    # the resource profile via time-expanded peak analysis).
    w_cp = 1.0
    w_cone = 0.30
    w_res = 0.20
    # Blend flow-based bottleneck with simple ratios
    res_signal = 0.4 * norm_bs + 0.2 * norm_dwb + 0.2 * norm_mr + 0.2 * norm_sr
    score = w_cp * norm_cp + w_cone * norm_cone + w_res * res_signal

    return score
# EVOLVE-BLOCK-END
