"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    Combines a feature-matrix + Simulated Annealing + coordinate-descent optimizer
    with a critical-ratio emphasis and a smooth soft-max surrogate feature.
    """
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    cycle_time = instance["cycle_time"]
    n = len(times)

    # Build successor graph from predecessors
    pred = instance["predecessors"]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)

    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1

    queue = [i for i in range(n) if in_degree[i] == 0]
    top_order = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        top_order.append(u)
        for v in succ[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Compute features in reverse topological order
    dist_down = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            max_succ = max(max_succ, dist_down[v])
        dist_down[u] = times[u] + max_succ

    dist_up = np.zeros(n, dtype=float)
    for u in top_order:
        max_pred = 0.0
        for p in pred[u]:
            max_pred = max(max_pred, dist_up[p])
        dist_up[u] = times[u] + max_pred

    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)

    total_succ = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0
        for v in succ[u]:
            s += 1 + total_succ[v]
        total_succ[u] = s

    total_pred = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0
        for p in pred[u]:
            s += 1 + total_pred[p]
        total_pred[u] = s

    top_rank = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        top_rank[u] = idx

    slack = cycle_time - dist_up - dist_down + times

    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist_down[i] > 1e-12:
            cpd[i] = times[i] / dist_down[i]
        else:
            cpd[i] = 0.0

    cpd_fanout = cpd * (1.0 + n_succ)

    total_downstream_time = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += times[v] + total_downstream_time[v]
        total_downstream_time[u] = s

    # Critical ratio: downstream workload relative to task duration
    # (high ratio = bottleneck that causes idle time in successors)
    eps = 1e-6
    critical_ratio = total_downstream_time / (times + eps)
    critical_ratio_fanout = critical_ratio * (1.0 + n_succ)

    # Smooth soft-max surrogate: log-sum-exp of critical_ratio and
    # normalized downstream workload. This is a smooth surrogate for the
    # nonsmooth max() that the greedy filling implicitly uses, helping the
    # optimizer navigate a smoother landscape.
    def soft_max_two(a, b, tau=1.0):
        m = np.maximum(a, b)
        return m + tau * np.log(np.exp((a - m) / tau) + np.exp((b - m) / tau))

    # Work in raw (unnormalized) space for the soft-max, then normalize.
    soft_crit_down = soft_max_two(critical_ratio, total_downstream_time, tau=1.0)

    # New feature: "bottleneck score" - how much downstream work depends on this task
    bottleneck_score = total_downstream_time / (cycle_time + 1e-9)

    # Position criticality
    position_crit = dist_down * (1.0 + n_succ)

    # Upstream dependency load
    upstream_load = total_pred * (1.0 + n_pred)

    # Normalize features
    def normalize(x):
        x = np.asarray(x, dtype=float)
        mn = x.min()
        mx = x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # Build feature matrix. Critical-ratio features are placed prominently.
    features = np.column_stack([
        normalize(critical_ratio),
        normalize(critical_ratio_fanout),
        normalize(total_downstream_time),
        normalize(soft_crit_down),
        normalize(dist_down),
        normalize(dist_up),
        normalize(times),
        normalize(n_succ),
        normalize(n_pred),
        normalize(total_succ),
        normalize(total_pred),
        1.0 - normalize(top_rank),
        1.0 - normalize(slack),
        normalize(cpd),
        normalize(cpd_fanout),
        normalize(bottleneck_score),
        normalize(position_crit),
        normalize(upstream_load),
    ])

    n_features = features.shape[1]

    # Fast simulation using heap for eligible tasks
    import heapq

    def simulate(scores):
        """Simulate greedy station filling and return number of stations."""
        remaining_count = n
        assigned_set = set()
        stations = 0
        pred_remaining = [len(pred[i]) for i in range(n)]

        heap = []
        for i in range(n):
            if pred_remaining[i] == 0:
                heapq.heappush(heap, (-scores[i], i))

        while remaining_count > 0:
            eligible = []
            temp_heap = []
            while heap:
                neg_score, task = heapq.heappop(heap)
                if task not in assigned_set:
                    eligible.append((neg_score, task))
                else:
                    temp_heap.append((neg_score, task))

            for item in temp_heap:
                heapq.heappush(heap, item)

            if not eligible:
                break

            eligible.sort()

            station_load = 0.0
            to_assign = []
            for neg_score, task in eligible:
                if station_load + times[task] <= cycle_time:
                    to_assign.append(task)
                    station_load += times[task]

            if not to_assign:
                to_assign = [eligible[0][1]]

            for task in to_assign:
                assigned_set.add(task)
                remaining_count -= 1
                for v in succ[task]:
                    pred_remaining[v] -= 1
                    if pred_remaining[v] == 0:
                        heapq.heappush(heap, (-scores[v], v))

            stations += 1

        return stations

    # Initial weights: emphasize critical-ratio / downstream features (measured
    # improvement driver), with the soft-max surrogate as a smoothing channel.
    # Feature order:
    # 0 critical_ratio, 1 critical_ratio_fanout, 2 total_downstream, 3 soft_crit_down,
    # 4 dist_down, 5 dist_up, 6 times, 7 n_succ, 8 n_pred, 9 total_succ, 10 total_pred,
    # 11 top_rank_inv, 12 slack_inv, 13 cpd, 14 cpd_fanout, 15 bottleneck, 16 position_crit, 17 upstream_load
    weights = np.array([0.20, 0.10, 0.16, 0.08, 0.12, 0.04, 0.10, 0.03, 0.03,
                        0.03, 0.03, 0.02, 0.02, 0.04, 0.04, 0.04, 0.04, 0.02])
    weights = weights / weights.sum()

    best_stations = simulate(features @ weights)
    best_weights = weights.copy()

    # Simulated Annealing
    np.random.seed(42)

    T0 = 1.0
    T_end = 0.005
    max_iters = 15000
    current_stations = best_stations
    current_weights = best_weights.copy()

    for i in range(max_iters):
        if time.time() - start_time > DEADLINE:
            break

        progress = i / max_iters
        noise_scale = 0.05 * (1.0 - progress) + 0.005

        new_w = current_weights.copy()
        noise = np.random.normal(0, noise_scale, n_features)
        new_w += noise
        new_w = np.clip(new_w, 0, None)
        if new_w.sum() > 1e-9:
            new_w = new_w / new_w.sum()
        else:
            new_w = current_weights

        score_vec = features @ new_w
        new_stations = simulate(score_vec)

        if new_stations < current_stations:
            current_stations = new_stations
            current_weights = new_w
            if new_stations < best_stations:
                best_stations = new_stations
                best_weights = new_w.copy()
        else:
            delta = new_stations - current_stations
            if delta == 0:
                if np.random.rand() < 0.5:
                    current_weights = new_w
            else:
                T = T0 * (T_end / T0) ** progress
                prob = math.exp(-delta / (T + 1e-9))
                if np.random.rand() < prob:
                    current_stations = new_stations
                    current_weights = new_w

    # Multi-stage coordinate descent polish
    step_sizes = [0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

    for step_size in step_sizes:
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for coord in range(n_features):
                if time.time() - start_time > DEADLINE:
                    break

                best_local = best_stations
                best_local_w = best_weights.copy()

                for delta in [step_size, -step_size, 2 * step_size, -2 * step_size]:
                    try_w = best_weights.copy()
                    try_w[coord] += delta
                    try_w = np.clip(try_w, 0, None)
                    if try_w.sum() < 1e-9:
                        continue
                    try_w = try_w / try_w.sum()

                    score_vec = features @ try_w
                    st = simulate(score_vec)

                    if st < best_local:
                        best_local = st
                        best_local_w = try_w.copy()

                if best_local < best_stations:
                    best_stations = best_local
                    best_weights = best_local_w
                    improved = True

        if time.time() - start_time > DEADLINE:
            break

    # Final high-precision polish pass with tighter tolerance
    if time.time() - start_time < DEADLINE - 10:
        for coord in range(n_features):
            if time.time() - start_time > DEADLINE - 5:
                break
            lo = max(0.0, best_weights[coord] - 0.001)
            hi = best_weights[coord] + 0.001
            best_hi = best_stations
            best_hi_w = best_weights.copy()
            # Fine sweep
            for t in np.linspace(lo, hi, 9):
                try_w = best_weights.copy()
                try_w[coord] = t
                try_w = np.clip(try_w, 0, None)
                if try_w.sum() < 1e-9:
                    continue
                try_w = try_w / try_w.sum()
                st = simulate(features @ try_w)
                if st < best_hi:
                    best_hi = st
                    best_hi_w = try_w.copy()
            if best_hi < best_stations:
                best_stations = best_hi
                best_weights = best_hi_w

    # Final check: try known good heuristics
    heuristic_candidates = [
        normalize(critical_ratio),
        normalize(critical_ratio_fanout),
        0.5 * normalize(critical_ratio) + 0.3 * normalize(dist_down) + 0.2 * normalize(times),
        0.4 * normalize(critical_ratio_fanout) + 0.3 * normalize(dist_down) + 0.3 * normalize(times),
        0.5 * normalize(soft_crit_down) + 0.3 * normalize(critical_ratio) + 0.2 * normalize(times),
        0.6 * normalize(dist_down) + 0.4 * normalize(times),
        0.5 * normalize(cpd) + 0.5 * normalize(cpd_fanout),
        0.5 * normalize(bottleneck_score) + 0.3 * normalize(dist_down) + 0.2 * normalize(times),
        0.4 * normalize(position_crit) + 0.3 * normalize(dist_down) + 0.3 * normalize(critical_ratio),
    ]

    final_scores = features @ best_weights
    for h in heuristic_candidates:
        s = simulate(h)
        if s < best_stations:
            best_stations = s
            final_scores = h

    # Add tiny tie-breaker
    final_scores = final_scores + 1e-8 * times

    return final_scores
# EVOLVE-BLOCK-END
