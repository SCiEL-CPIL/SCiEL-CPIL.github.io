"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def _greedy_assign(times, preds, cycle_time, priority_scores, succ=None, remaining_preds_init=None):
    """Simulate the greedy station filling with given priority scores."""
    n = len(times)
    if n == 0:
        return 0

    if succ is None:
        succ = [[] for _ in range(n)]
        for i in range(n):
            for p in preds[i]:
                succ[p].append(i)

    if remaining_preds_init is None:
        remaining_preds_init = [len(preds[i]) for i in range(n)]

    remaining_preds = remaining_preds_init[:]
    eligible = [i for i in range(n) if remaining_preds[i] == 0]
    in_eligible = [False] * n
    for i in eligible:
        in_eligible[i] = True

    stations = 0
    remaining = n
    while remaining > 0:
        stations += 1
        load = 0.0
        while True:
            best_task = -1
            best_prio = -1e18
            for i in eligible:
                if times[i] <= cycle_time - load + 1e-9:
                    p = priority_scores[i]
                    if p > best_prio:
                        best_prio = p
                        best_task = i
            if best_task == -1:
                break
            load += times[best_task]
            remaining -= 1
            # Remove from eligible list efficiently
            in_eligible[best_task] = False
            # Find index and swap-remove
            for idx, t in enumerate(eligible):
                if t == best_task:
                    eligible[idx] = eligible[-1]
                    eligible.pop()
                    break
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible.append(s)
                    in_eligible[s] = True

    return stations


def _build_graph(preds, n):
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    remaining_preds_init = [len(preds[i]) for i in range(n)]
    return succ, remaining_preds_init


def _compute_features(times, preds, n, succ, remaining_preds_init):
    """Compute multiple priority features."""
    # Topological order
    in_degree = remaining_preds_init[:]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)

    # Bottom-level rank (longest path from task to end)
    bottom_level = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for s in succ[node]:
            if bottom_level[s] > max_succ:
                max_succ = bottom_level[s]
        bottom_level[node] = times[node] + max_succ

    # Top-level rank (longest path from start to task)
    top_level = np.zeros(n, dtype=float)
    for i in range(len(topo_order)):
        node = topo_order[i]
        max_pred = 0.0
        for p in preds[node]:
            if top_level[p] + times[p] > max_pred:
                max_pred = top_level[p] + times[p]
        top_level[node] = max_pred

    # Number of successors
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)

    # Number of predecessors
    num_pred = np.array(remaining_preds_init, dtype=float)

    # Position in topological order (earlier = higher priority for some heuristics)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx

    # Critical path position: normalize bottom_level relative to total
    total_bl = bottom_level.max() if n > 0 else 1.0
    if total_bl < 1e-9:
        total_bl = 1.0

    return {
        'bottom_level': bottom_level,
        'top_level': top_level,
        'num_succ': num_succ,
        'num_pred': num_pred,
        'pos_in_topo': pos_in_topo,
        'times': times.copy(),
    }


def _normalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def priority(instance):
    start_time = time.time()
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)

    if n == 0:
        return np.array([])

    succ, remaining_preds_init = _build_graph(preds, n)
    features = _compute_features(times, preds, n, succ, remaining_preds_init)

    bl_norm = _normalize(features['bottom_level'])
    t_norm = _normalize(features['times'])
    ns_norm = _normalize(features['num_succ'])
    tl_norm = _normalize(features['top_level'])
    np_norm = _normalize(features['num_pred'])
    pos_norm = _normalize(features['pos_in_topo'])

    # Initial priority: weighted combination
    # Emphasize bottom-level (critical path to end) and time
    score = 1.0 * bl_norm + 0.5 * t_norm + 0.3 * ns_norm + 0.2 * tl_norm

    best_score = score.copy()
    best_stations = _greedy_assign(times, preds, cycle_time, score, succ, remaining_preds_init)

    # Try several weight combinations
    weight_combos = [
        (1.0, 0.5, 0.3, 0.2, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 0.5, 0.5, 0.5, 0.0),
        (2.0, 0.5, 0.5, 0.5, 0.0),
        (1.0, 0.2, 0.2, 0.2, 0.2),
        (1.5, 0.8, 0.4, 0.3, 0.1),
        (1.0, 0.7, 0.3, 0.3, 0.1),
        (1.2, 0.6, 0.5, 0.2, 0.1),
        (0.5, 1.0, 0.5, 0.5, 0.0),
        (1.0, 0.3, 0.7, 0.3, 0.0),
        (1.5, 0.5, 0.3, 0.5, 0.0),
        (0.8, 0.8, 0.8, 0.8, 0.0),
        (1.0, 0.5, 0.0, 0.5, 0.5),
        (1.3, 0.7, 0.4, 0.4, 0.1),
    ]

    for w_bl, w_t, w_ns, w_tl, w_np in weight_combos:
        if time.time() - start_time > _DEADLINE:
            break
        s = w_bl * bl_norm + w_t * t_norm + w_ns * ns_norm + w_tl * tl_norm + w_np * np_norm
        st = _greedy_assign(times, preds, cycle_time, s, succ, remaining_preds_init)
        if st < best_stations:
            best_stations = st
            best_score = s

    # Hill climbing: perturb individual task priorities
    rng = np.random.default_rng(42)
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() - start_time > _DEADLINE:
            break

        # Occasionally do multi-task perturbation for larger jumps
        if rng.random() < 0.2:
            # Perturb 2-3 random tasks
            num_perturb = int(rng.integers(2, 4))
            task_indices = rng.choice(n, size=num_perturb, replace=False)
            trial_score = best_score.copy()
            for ti in task_indices:
                trial_score[ti] += rng.normal(0, 0.3)
        else:
            # Single task perturbation with adaptive step size
            task_idx = rng.integers(0, n)
            step_size = 0.15 if iteration < max_iter // 2 else 0.08
            perturbation = rng.normal(0, step_size)
            trial_score = best_score.copy()
            trial_score[task_idx] += perturbation

        st = _greedy_assign(times, preds, cycle_time, trial_score, succ, remaining_preds_init)
        if st < best_stations:
            best_stations = st
            best_score = trial_score
        elif st == best_stations:
            # Accept with some probability to escape local optima
            if rng.random() < 0.15:
                best_score = trial_score

    # Final polish: fine-tune around best solution
    if time.time() - start_time < _DEADLINE - 5:
        for iteration in range(100):
            if time.time() - start_time > _DEADLINE:
                break
            task_idx = rng.integers(0, n)
            perturbation = rng.normal(0, 0.05)
            trial_score = best_score.copy()
            trial_score[task_idx] += perturbation

            st = _greedy_assign(times, preds, cycle_time, trial_score, succ, remaining_preds_init)
            if st < best_stations:
                best_stations = st
                best_score = trial_score

    return best_score
# EVOLVE-BLOCK-END
