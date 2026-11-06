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

    # Build successor list
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

    # NEW: Compute earliest start time (longest path from sources) for each task
    # earliest_start[i] = max over predecessors p of (earliest_start[p] + times[p]), or 0 if no preds
    # This is the earliest time task i can begin, assuming all predecessors are done as early as possible.
    earliest_start = np.zeros(n)
    for u in topo_order:
        if pred_list[u]:
            earliest_start[u] = max(earliest_start[p] + times[p] for p in pred_list[u])
        else:
            earliest_start[u] = 0.0

    # Positional criticality: ratio of earliest start to cycle time
    # Tasks with high earliest_start are "late-start" tasks that are constrained by long predecessor chains.
    # However, the directive says to prioritize tasks that "strictly constrain the timeline's start phase first."
    # Re-reading: "ratio of their earliest possible start time to the cycle time, to identify and place tasks
    # that strictly constrain the timeline's start phase first."
    # This is ambiguous. Let me think: tasks with HIGH earliest_start are those that can't start until late.
    # But "constrain the start phase" might mean tasks that, if placed late, delay the entire schedule.
    # Actually, I think the intent is: tasks with high earliest_start are on critical paths and should be
    # prioritized so they don't become bottlenecks. Let me use earliest_start / cycle_time as a signal.
    #
    # Actually, re-reading more carefully: "positional criticality, defined as the ratio of their earliest
    # possible start time (longest path from sources) to the cycle time, to identify and place tasks that
    # strictly constrain the timeline's start phase first."
    #
    # I interpret this as: tasks with high earliest_start are positionally critical because they represent
    # the temporal depth of the precedence graph. Placing them early (when they become eligible) ensures
    # the timeline progresses. So we should give them HIGH priority.
    #
    # But wait - if a task has a high earliest_start, it means it has many predecessors. In the greedy
    # packing, it won't be eligible until all predecessors are placed. So its priority only matters
    # once it's eligible. The key is: among eligible tasks, which should go first?
    #
    # A task with high earliest_start that is now eligible means its predecessors were all placed quickly,
    # so placing it now keeps the critical path moving. This is a good signal.

    ct = max(cycle_time, 1.0)
    positional_crit = earliest_start / ct

    # Also compute latest finish time (longest path from task to sinks) for symmetry
    # latest_finish[i] = times[i] + max over successors s of (latest_finish[s])
    latest_finish = np.zeros(n)
    for u in reversed(topo_order):
        if successors[u]:
            latest_finish[u] = times[u] + max(latest_finish[s] for s in successors[u])
        else:
            latest_finish[u] = times[u]

    # Total makespan lower bound = max(latest_finish)
    # Slack of task i = (makespan_lower_bound - earliest_start[i] - times[i])
    # Low slack = critical path task
    makespan_lb = np.max(latest_finish) if n > 0 else 1.0
    slack = makespan_lb - earliest_start - times

    # Normalize all metrics
    max_succ_time = np.max(succ_time_sum) if np.max(succ_time_sum) > 0 else 1.0
    max_pred_time = np.max(pred_time_sum) if np.max(pred_time_sum) > 0 else 1.0
    max_succ_count = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    max_pred_count = np.max(pred_count) if np.max(pred_count) > 0 else 1.0
    max_t = np.max(times) if np.max(times) > 0 else 1.0
    max_pos_crit = np.max(positional_crit) if np.max(positional_crit) > 0 else 1.0
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0

    # Weights: emphasize successor-based metrics (proven effective) + add positional criticality
    # and low slack (critical path)
    w_succ_time = 2.0
    w_pred_time = 0.8
    w_succ_count = 1.5
    w_pred_count = 0.4
    w_time = 0.5
    w_pos_crit = 1.5
    w_low_slack = 1.0

    # For slack, we want LOW slack to have HIGH priority, so we invert
    # Use (max_slack - slack) / max_slack so low slack -> high value
    inverted_slack = (max_slack - slack) / max_slack if max_slack > 0 else np.zeros(n)

    scores = (w_succ_time * (succ_time_sum / max_succ_time)
              + w_pred_time * (pred_time_sum / max_pred_time)
              + w_succ_count * (succ_count / max_succ_count)
              + w_pred_count * (pred_count / max_pred_count)
              + w_time * (times / max_t)
              + w_pos_crit * (positional_crit / max_pos_crit)
              + w_low_slack * inverted_slack)

    # Deadline guard
    if time.time() > deadline - safety_margin:
        pass  # We're already done computing, just return

    return scores
# EVOLVE-BLOCK-END
