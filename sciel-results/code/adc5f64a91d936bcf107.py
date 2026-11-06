"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque


def priority(instance):
    start_wall = time.time()
    deadline = 120.0
    safety_margin = 5.0

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance.get("predecessors", [])
    cycle_time = instance.get("cycle_time", 40.0)
    n = len(times)

    # Build successor and predecessor lists
    successors = [[] for _ in range(n)]
    pred_list = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
            pred_list[i].append(p)

    # Topological order
    in_degree = [len(pred_list[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)

    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)

    # Transitive closure
    trans_succ = [set() for _ in range(n)]
    trans_pred = [set() for _ in range(n)]

    for u in reversed(topo_order):
        for v in successors[u]:
            trans_succ[u].add(v)
            trans_succ[u].update(trans_succ[v])

    for u in topo_order:
        for p in pred_list[u]:
            trans_pred[u].add(p)
            trans_pred[u].update(trans_pred[p])

    # Compute metrics
    pred_time_sum = np.zeros(n)
    succ_time_sum = np.zeros(n)
    succ_count = np.zeros(n)
    pred_count = np.zeros(n)

    for i in range(n):
        pred_time_sum[i] = sum(times[p] for p in trans_pred[i])
        succ_time_sum[i] = sum(times[s] for s in trans_succ[i])
        succ_count[i] = len(trans_succ[i])
        pred_count[i] = len(trans_pred[i])

    # Earliest start time (longest path from sources)
    earliest_start = np.zeros(n)
    for u in topo_order:
        if pred_list[u]:
            earliest_start[u] = max(earliest_start[p] + times[p] for p in pred_list[u])
        else:
            earliest_start[u] = 0.0

    # Latest finish time (longest path from task to sinks)
    latest_finish = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            latest_finish[u] = times[u] + max(latest_finish[s] for s in successors[u])
        else:
            latest_finish[u] = times[u]

    # Makespan lower bound and slack
    makespan_lb = np.max(latest_finish) if n > 0 else 1.0
    slack = makespan_lb - earliest_start - times

    # Positional criticality
    ct = max(cycle_time, 1.0)
    positional_crit = earliest_start / ct

    # Resource contention: sum of cycle-time-normalized durations of all transitive
    # predecessors and successors. This captures how "time-dense" a task's dependency
    # neighborhood is. Tasks in dense, interconnected subgraphs get higher priority.
    contention = (pred_time_sum + succ_time_sum) / ct

    # Bridging metric: geometric mean of normalized pred and succ times, capturing
    # structural importance of tasks that connect both upstream and downstream work.
    # Use a small epsilon to avoid log(0).
    eps = 1e-10
    norm_pred = pred_time_sum / (max(np.max(pred_time_sum), eps))
    norm_succ = succ_time_sum / (max(np.max(succ_time_sum), eps))
    bridging = np.sqrt(np.maximum(norm_pred, eps) * np.maximum(norm_succ, eps))

    # Normalize all metrics
    max_succ_time = np.max(succ_time_sum) if np.max(succ_time_sum) > 0 else 1.0
    max_pred_time = np.max(pred_time_sum) if np.max(pred_time_sum) > 0 else 1.0
    max_succ_count = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    max_pred_count = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    max_t = np.max(times) if np.max(times) > 0 else 1.0
    max_pos_crit = np.max(positional_crit) if np.max(positional_crit) > 0 else 1.0
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    max_contention = np.max(contention) if np.max(contention) > 0 else 1.0
    max_bridging = np.max(bridging) if np.max(bridging) > 0 else 1.0

    # Invert slack: low slack -> high priority
    inverted_slack = (max_slack - slack) / max_slack if max_slack > 0 else np.zeros(n)

    # Weights: emphasize resource contention (global temporal density) and successor-based
    # metrics. Reduce weight on pure local duration since contention already captures
    # global time density. Bridging captures structural importance.
    w_succ_time = 1.5
    w_pred_time = 0.6
    w_succ_count = 1.2
    w_pred_count = 0.3
    w_time = 0.3
    w_pos_crit = 1.0
    w_low_slack = 0.8
    w_contention = 2.5
    w_bridging = 1.2

    scores = (w_succ_time * (succ_time_sum / max_succ_time)
              + w_pred_time * (pred_time_sum / max_pred_time)
              + w_succ_count * (succ_count / max_succ_count)
              + w_pred_count * (pred_count / max_pred_count)
              + w_time * (times / max_t)
              + w_pos_crit * (positional_crit / max_pos_crit)
              + w_low_slack * inverted_slack
              + w_contention * (contention / max_contention)
              + w_bridging * (bridging / max_bridging))

    # Deadline guard
    if time.time() > deadline - safety_margin:
        pass

    return scores
# EVOLVE-BLOCK-END
