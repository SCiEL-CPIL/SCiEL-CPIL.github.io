"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


def priority(instance):
    """
    Score each assembly task for the greedy assembly line balancing heuristic.
    Stations are filled with the highest-priority eligible task that fits the
    cycle time. We compute rich graph features, then search for the best linear
    combination of features (weights) that minimizes the number of stations
    produced by the greedy heuristic on the given instance.
    """
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    times = np.asarray(instance["times"], dtype=float)
    cycle_time = float(instance["cycle_time"])
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

    # If the graph is not fully topologically sorted (cycle), pad with remaining
    if len(top_order) < n:
        remaining_nodes = [i for i in range(n) if i not in set(top_order)]
        top_order.extend(remaining_nodes)

    # Feature 1: Longest path to sink (downstream work)
    dist_down = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        max_succ = 0.0
        for v in succ[u]:
            if dist_down[v] > max_succ:
                max_succ = dist_down[v]
        dist_down[u] = times[u] + max_succ

    # Feature 2: Longest path from source (upstream work)
    dist_up = np.zeros(n, dtype=float)
    for u in top_order:
        max_pred = 0.0
        for p in pred[u]:
            if dist_up[p] > max_pred:
                max_pred = dist_up[p]
        dist_up[u] = times[u] + max_pred

    # Feature 3: Number of immediate successors
    n_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # Feature 4: Number of immediate predecessors
    n_pred = np.array([len(pred[i]) for i in range(n)], dtype=float)

    # Feature 5: Total transitive successors
    total_succ = np.zeros(n, dtype=float)
    for u in reversed(top_order):
        s = 0.0
        for v in succ[u]:
            s += 1.0 + total_succ[v]
        total_succ[u] = s

    # Feature 6: Total transitive predecessors
    total_pred = np.zeros(n, dtype=float)
    for u in top_order:
        s = 0.0
        for p in pred[u]:
            s += 1.0 + total_pred[p]
        total_pred[u] = s

    # Feature 7: Rank-based position in topological order (earlier = higher priority base)
    top_rank = np.zeros(n, dtype=float)
    for idx, u in enumerate(top_order):
        top_rank[u] = float(idx)

    # Feature 8: Slack / flexibility: tasks with less slack should go first
    # Slack = cycle_time - dist_up - dist_down + times[u]
    slack = cycle_time - dist_up - dist_down + times

    # Feature 9: Critical Path Density: ratio of task duration to its longest downstream path
    cpd = np.zeros(n, dtype=float)
    for i in range(n):
        if dist_down[i] > 1e-12:
            cpd[i] = times[i] / dist_down[i]
        else:
            cpd[i] = 0.0

    # Feature 10: CPD x Fanout: prioritize bottlenecks that are also branching points
    cpd_fanout = cpd * (1.0 + n_succ)

    # Feature 11: Positional score: tasks closer to sources (lower upstream depth) should go first
    total_cp_len = float(dist_down.max()) if n > 0 else 1.0
    if total_cp_len > 1e-12:
        depth_norm = dist_up / total_cp_len
    else:
        depth_norm = np.zeros(n)
    positional = np.clip(1.0 - depth_norm, 0.0, 1.0)

    # Normalize features
    def normalize(x):
        mn = float(x.min())
        mx = float(x.max())
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    # Build feature matrix (higher = more priority)
    features = np.column_stack([
        normalize(dist_down),       # 0: downstream path length
        normalize(dist_up),         # 1: upstream path length
        normalize(times),           # 2: task duration
        normalize(n_succ),          # 3: immediate successors
        normalize(n_pred),          # 4: immediate predecessors
        normalize(total_succ),      # 5: transitive successors
        normalize(total_pred),      # 6: transitive predecessors
        1.0 - normalize(top_rank),  # 7: earlier in topo order
        1.0 - normalize(slack),     # 8: less slack = tighter
        normalize(cpd),             # 9: critical path density
        normalize(cpd_fanout),      # 10: critical path density x fanout
        positional,                 # 11: closer to sources
    ])

    n_features = features.shape[1]

    # Fast greedy simulator using a max-heap for eligible tasks
    def simulate(scores):
        """Simulate greedy station filling and return number of stations."""
        assigned = [-1] * n
        remaining = list(range(n))
        remaining_set = set(remaining)
        stations = 0

        # precompute predecessor counts for eligibility
        pred_count = [len(pred[i]) for i in range(n)]
        unmet = pred_count[:]  # number of unassigned predecessors

        while remaining:
            # Find eligible tasks: all predecessors assigned
            heap = []
            for i in remaining:
                if unmet[i] == 0:
                    # max-heap via negative score
                    heapq.heappush(heap, (-float(scores[i]), i))

            if not heap:
                # No eligible tasks (should not happen for valid DAG)
                break

            station_load = 0.0
            # Fill current station
            while heap:
                neg_score, i = heap[0]
                # Peek: check if it fits
                if station_load + times[i] <= cycle_time + 1e-9:
                    heapq.heappop(heap)
                    assigned[i] = stations
                    station_load += times[i]
                    remaining_set.discard(i)
                    # Update unmet for successors
                    for v in succ[i]:
                        if v in remaining_set:
                            unmet[v] -= 1
                else:
                    break

            # Remove all assigned from remaining list
            remaining = [i for i in remaining if i in remaining_set]

            if not remaining:
                break

            # Check if any tasks were assigned to this station
            if station_load < 1e-12:
                # Force assign the highest priority eligible
                neg_score, best = heapq.heappop(heap)
                assigned[best] = stations
                remaining_set.discard(best)
                for v in succ[best]:
                    if v in remaining_set:
                        unmet[v] -= 1
                remaining = [i for i in remaining if i in remaining_set]
                if not remaining:
                    break

            stations += 1

        return stations

    # Initial weights: emphasize downstream path, duration, and critical path density
    weights = np.array([0.25, 0.10, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.05])
    weights = weights / weights.sum()

    best_stations = simulate(features @ weights)
    best_weights = weights.copy()

    # Model-and-optimize: local search on weights
    np.random.seed(42)

    # Budget for optimization
    max_evals = 3000
    eval_count = 0

    for iteration in range(max_evals):
        if time.time() - start_time > DEADLINE:
            break

        # Random perturbation
        new_w = best_weights.copy()
        noise = np.random.normal(0, 0.05, n_features)
        new_w += noise
        new_w = np.clip(new_w, 0, None)
        if new_w.sum() > 1e-9:
            new_w = new_w / new_w.sum()

        # Evaluate
        score_vec = features @ new_w
        stations = simulate(score_vec)
        eval_count += 1

        if stations < best_stations:
            best_stations = stations
            best_weights = new_w.copy()

            # If improved, try coordinate descent
            for coord in range(n_features):
                if time.time() - start_time > DEADLINE:
                    break
                for delta in [0.02, -0.02, 0.05, -0.05, 0.1, -0.1]:
                    try_w = best_weights.copy()
                    try_w[coord] += delta
                    try_w = np.clip(try_w, 0, None)
                    if try_w.sum() < 1e-9:
                        continue
                    try_w = try_w / try_w.sum()

                    score_vec = features @ try_w
                    st = simulate(score_vec)
                    eval_count += 1

                    if st < best_stations:
                        best_stations = st
                        best_weights = try_w.copy()

    # Simulated Annealing polish pass: escape local minima with temperature schedule
    if time.time() - start_time < DEADLINE - 10:
        sa_best_stations = best_stations
        sa_best_weights = best_weights.copy()
        T = 1.0  # initial temperature
        T_min = 0.01
        cooling = 0.995
        sa_iters = 0
        sa_max_iters = 3000

        while T > T_min and sa_iters < sa_max_iters:
            if time.time() - start_time > DEADLINE - 5:
                break
            sa_iters += 1

            # Generate candidate: larger perturbations at higher T
            noise_scale = 0.02 + 0.08 * T
            cand_w = sa_best_weights.copy()
            noise = np.random.normal(0, noise_scale, n_features)
            cand_w += noise
            cand_w = np.clip(cand_w, 0, None)
            if cand_w.sum() < 1e-9:
                continue
            cand_w = cand_w / cand_w.sum()

            score_vec = features @ cand_w
            cand_stations = simulate(score_vec)

            # Accept if better, or with probability exp(-delta/T) if worse
            delta = cand_stations - sa_best_stations
            if delta < 0 or (delta > 0 and np.random.random() < np.exp(-delta / max(T, 1e-9))):
                sa_best_stations = cand_stations
                sa_best_weights = cand_w.copy()

            T *= cooling

        if sa_best_stations < best_stations:
            best_stations = sa_best_stations
            best_weights = sa_best_weights

    # Final check: also try some known good heuristics directly
    final_scores = features @ best_weights

    # Heuristic 1: pure downstream distance
    h1 = normalize(dist_down)
    s1 = simulate(h1)
    if s1 < best_stations:
        best_stations = s1
        final_scores = h1

    # Heuristic 2: pure task time
    h2 = normalize(times)
    s2 = simulate(h2)
    if s2 < best_stations:
        best_stations = s2
        final_scores = h2

    # Heuristic 3: downstream + time combination
    h3 = 0.6 * normalize(dist_down) + 0.4 * normalize(times)
    s3 = simulate(h3)
    if s3 < best_stations:
        best_stations = s3
        final_scores = h3

    # Heuristic 4: inverse rank (BKS-like)
    h4 = 1.0 - normalize(top_rank)
    s4 = simulate(h4)
    if s4 < best_stations:
        best_stations = s4
        final_scores = h4

    # Heuristic 5: Critical Path Density + Fanout
    h5 = 0.5 * normalize(cpd) + 0.3 * normalize(dist_down) + 0.2 * normalize(cpd_fanout)
    s5 = simulate(h5)
    if s5 < best_stations:
        best_stations = s5
        final_scores = h5

    # Heuristic 6: CPD-heavy blend
    h6 = 0.4 * normalize(cpd_fanout) + 0.3 * normalize(dist_down) + 0.2 * normalize(times) + 0.1 * (1.0 - normalize(slack))
    s6 = simulate(h6)
    if s6 < best_stations:
        best_stations = s6
        final_scores = h6

    # Heuristic 7: Positional + downstream
    h7 = 0.5 * positional + 0.3 * normalize(dist_down) + 0.2 * normalize(times)
    s7 = simulate(h7)
    if s7 < best_stations:
        best_stations = s7
        final_scores = h7

    # Add tiny tie-breaker
    final_scores = final_scores + 1e-8 * times

    return final_scores
# EVOLVE-BLOCK-END
