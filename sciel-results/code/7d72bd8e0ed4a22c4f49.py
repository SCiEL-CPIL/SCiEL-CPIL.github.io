"""Assembly Line Balancing - Model and Optimize approach (combined)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque


def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Greedy station filling: repeatedly pick the highest-priority eligible task that fits.
    Returns number of stations.
    """
    n = len(times)
    if n == 0:
        return 0

    successors = defaultdict(list)
    unassigned_preds = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)

    eligible = set(i for i in range(n) if unassigned_preds[i] == 0)
    if n > 0 and not eligible:
        return 10**9

    # Pre-check for infeasible task
    if np.any(times > cycle_time):
        return 10**9

    station_count = 1
    current_load = 0.0

    while eligible:
        # Pick highest-priority eligible task that fits
        candidates = sorted(eligible, key=lambda i: -priority_scores[i])
        placed = False
        for task in candidates:
            if times[task] <= cycle_time - current_load + 1e-9:
                current_load += times[task]
                eligible.discard(task)
                placed = True
                for succ in successors[task]:
                    unassigned_preds[succ] -= 1
                    if unassigned_preds[succ] == 0:
                        eligible.add(succ)
                break
        if not placed:
            station_count += 1
            current_load = 0.0

    return station_count


def _compute_critical_path(times, predecessors, successors, n):
    """Longest path from each task to any sink (including its own time)."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    if len(topo_order) != n:
        return None
    cp = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        cp[u] = times[u]
        for v in successors[u]:
            cp[u] = max(cp[u], times[u] + cp[v])
    return cp


def _compute_downstream_work(times, predecessors, successors, n):
    """Total work (sum of times) in the transitive successor subtree of each task."""
    in_degree = np.array([len(predecessors[i]) for i in range(n)], dtype=int)
    in_deg_copy = in_degree.copy()
    q = deque()
    for i in range(n):
        if in_deg_copy[i] == 0:
            q.append(i)
    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
    if len(topo_order) != n:
        return None
    dw = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        s = 0.0
        for v in successors[u]:
            s += times[v] + dw[v]
        dw[u] = s
    return dw


def _count_transitive(start, adj):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for s in adj[node]:
            if s not in visited:
                visited.add(s)
                stack.append(s)
    return len(visited)


def _build_candidates(times, predecessors, successors, pred_map, cycle_time, n):
    """Build a diverse set of initial priority vectors (richer feature mix)."""
    cp = _compute_critical_path(times, predecessors, successors, n)
    if cp is None:
        cp = np.zeros(n, dtype=float)
    dw = _compute_downstream_work(times, predecessors, successors, n)
    if dw is None:
        dw = np.zeros(n, dtype=float)

    num_succ = np.array([_count_transitive(i, successors) for i in range(n)], dtype=float)
    num_pred = np.array([_count_transitive(i, pred_map) for i in range(n)], dtype=float)
    direct_succ = np.array([len(successors[i]) for i in range(n)], dtype=float)
    direct_pred = np.array([len(predecessors[i]) for i in range(n)], dtype=float)

    # Normalized features (downstream subtree weight is the strongest measured signal)
    def norm(a):
        mn, mx = a.min(), a.max()
        if mx - mn < 1e-12:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    cp_n = norm(cp)
    dw_n = norm(dw)
    num_succ_n = norm(num_succ)
    direct_succ_n = norm(direct_succ)
    times_n = norm(times)

    # Bottleneck slack ratio: small (cycle_time - time)/(1+branching) => high priority
    slack_ratio = (cycle_time - times) / (1.0 + direct_succ)
    slack_ratio_n = norm(slack_ratio)

    candidates = []
    # Critical path family
    candidates.append(cp.copy())
    candidates.append(cp + 0.1 * times)
    candidates.append(cp + 0.5 * num_succ)
    candidates.append(cp + 0.3 * direct_succ)
    # Downstream subtree weight family (measured strongest)
    candidates.append(dw.copy())
    candidates.append(dw + 0.2 * cp)
    candidates.append(dw + 0.3 * num_succ)
    # Time family
    candidates.append(times.copy())
    candidates.append(times + 0.2 * cp)
    # Successor family
    candidates.append(num_succ.copy())
    candidates.append(direct_succ.copy())
    candidates.append(cp + num_succ)
    # Combined normalized (emphasize downstream + critical path)
    candidates.append(0.5 * dw_n + 0.3 * cp_n + 0.2 * num_succ_n)
    candidates.append(0.6 * cp_n + 0.4 * direct_succ_n)
    candidates.append(0.4 * dw_n + 0.4 * cp_n + 0.2 * direct_succ_n)
    candidates.append(cp + times - 0.1 * num_pred)
    # Bottleneck slack ratio family
    candidates.append(-slack_ratio)
    candidates.append(-slack_ratio + 0.1 * cp)
    candidates.append(-slack_ratio_n + 0.2 * dw_n)
    return candidates, cp


def _local_search(priority, times, predecessors, cycle_time, n, max_iters, deadline, rng):
    """
    Local search over priority orderings via pairwise swaps.
    Uses best-acceptance: keep any improvement, track the best vector.
    """
    best_priority = priority.copy()
    best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)
    cur = priority.copy()
    cur_score = best_score

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        i, j = rng.sample(range(n), 2)
        cur[i], cur[j] = cur[j], cur[i]
        new_score = _greedy_balance(times, predecessors, cycle_time, cur)
        if new_score < best_score:
            best_score = new_score
            best_priority = cur.copy()
            cur_score = new_score
        else:
            # revert to keep the current neighborhood consistent
            cur[i], cur[j] = cur[j], cur[i]
        if best_score == 1:
            break
    return best_priority, best_score


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]

    n = len(times)
    if n == 0:
        return np.array([])

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Build adjacency
    successors = defaultdict(list)
    pred_map = defaultdict(list)
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            pred_map[i].append(pred)

    candidates, cp = _build_candidates(times, predecessors, successors, pred_map, cycle_time, n)

    # Evaluate initial candidates, keep best
    best_priority = None
    best_score = float('inf')
    for pri in candidates:
        if time.time() > deadline - 30:
            break
        score = _greedy_balance(times, predecessors, cycle_time, pri)
        if score < best_score:
            best_score = score
            best_priority = pri.copy()
        if best_score == 1:
            break

    if best_priority is None:
        best_priority = cp.copy()
        best_score = _greedy_balance(times, predecessors, cycle_time, best_priority)

    rng = np.random.RandomState(42)

    # Main local search from the best initial
    if time.time() < deadline - 10 and best_score > 1:
        best_priority, best_score = _local_search(
            best_priority, times, predecessors, cycle_time, n,
            max_iters=50000, deadline=deadline - 5, rng=rng
        )

    # Restart local search from a few other good starting points if time remains
    remaining = deadline - 5 - time.time()
    if remaining > 25 and best_score > 1:
        # pick the top distinct candidates as restart seeds
        order = sorted(range(len(candidates)),
                       key=lambda k: _greedy_balance(times, predecessors, cycle_time, candidates[k]))
        for k in order[:3]:
            if time.time() > deadline - 5:
                break
            pri = candidates[k].copy()
            pri = pri + rng.normal(0, 0.05, n)
            p, s = _local_search(
                pri, times, predecessors, cycle_time, n,
                max_iters=20000, deadline=deadline - 5, rng=rng
            )
            if s < best_score:
                best_score = s
                best_priority = p

    return best_priority
# EVOLVE-BLOCK-END
