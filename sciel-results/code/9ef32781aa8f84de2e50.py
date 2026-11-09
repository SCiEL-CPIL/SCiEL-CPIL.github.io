import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher priority (score) is scheduled first among eligible activities.

    Design (refined within the CPM + resource-aware family):
    1. Compute Critical Path Length (CPL): longest path from each activity to project end.
       This is the primary, dominant priority signal.
    2. Compute a resource-contention signal using log-sum-exp smooth max over demand/capacity ratios.
       This favors activities that use scarce resources, reducing later blocking.
    3. Compute a resource-weighted tail: longest path where each node is weighted by its
       resource intensity * duration. This captures chains of resource-heavy activities.
    4. Compute slack (LST - EST) to penalize flexible activities.
    5. Small tie-break by duration for determinism.

    The serial SGS benefits most from scheduling critical-path activities first;
    the resource terms are secondary signals to avoid wasting scarce capacity and
    to prioritize activities that unblock resource-intensive downstream work.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    preds = instance["predecessors"]
    n_res = len(capacities)

    # Build successor list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort (Kahn's algorithm)
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(topo_order) != n:
        # Fallback: just use durations as priority
        return durations.copy()

    # Compute Critical Path Length (CPL) - longest path from activity to project end
    cpl = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            cpl[u] = durations[u]
        else:
            cpl[u] = durations[u] + max(cpl[v] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute resource-contention signal using log-sum-exp smooth max over demand/capacity ratios.
    # This captures multi-resource pressure: activities with multiple moderately high ratios
    # get appropriately high scores (unlike hard max which ignores all but the single worst).
    tau = 0.1  # small enough to approximate max but smooth enough to capture tail
    res_ratio = np.zeros(n, dtype=float)
    for i in range(n):
        ratios = []
        for j in range(n_res):
            if capacities[j] > 0 and demands[i][j] > 0:
                ratios.append(demands[i][j] / capacities[j])
        if ratios:
            arr = np.asarray(ratios, dtype=float)
            m = arr.max()
            if m > 1e-10:
                res_ratio[i] = tau * np.log(np.sum(np.exp((arr - m) / tau))) + m
            else:
                res_ratio[i] = 0.0
        else:
            res_ratio[i] = 0.0

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute resource-weighted tail: longest path where each node contributes
    # (resource_intensity * duration). This captures how much resource-heavy work
    # lies ahead of each activity, including its own contribution.
    res_tail = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        own_weight = res_ratio[u] * durations[u]
        if not successors[u]:
            res_tail[u] = own_weight
        else:
            max_succ = max(res_tail[v] for v in successors[u])
            res_tail[u] = own_weight + max_succ

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Latest Start Time (LST) to derive slack
    project_duration = max(cpl)
    lst = np.zeros(n, dtype=float)
    for u in topo_order:
        if not successors[u]:
            lst[u] = project_duration - durations[u]
        else:
            lst[u] = min(lst[v] - durations[u] for v in successors[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Compute Earliest Start Time (EST) to derive slack
    est = np.zeros(n, dtype=float)
    for u in topo_order:
        if not preds[u]:
            est[u] = 0
        else:
            est[u] = max(est[p] + durations[p] for p in preds[u])

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl.copy()

    # Total Slack = LST - EST (higher slack = more flexibility)
    slack = lst - est

    # Normalize components
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    cpl_norm = normalize(cpl)
    res_norm = normalize(res_ratio)
    tail_norm = normalize(res_tail)
    dur_norm = normalize(durations)
    slack_norm = normalize(slack)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl_norm.copy()

    # Compute resource saturation depth: for each activity, count how many
    # downstream successors (including itself) also have high demand on the same
    # bottleneck resource. This captures "resource congestion chains" — if many
    # activities in a chain all need resource j, scheduling one early reduces
    # the probability that later ones get blocked.
    #
    # For efficiency, we use the top-2 resources per activity and count overlaps
    # along the critical path successors.
    # We approximate by: for each activity u, sum over its successors on the
    # critical path of (shared_bottleneck_count / total_downstream_count).
    #
    # Simpler proxy: count how many successors share the same max-demand resource.
    res_saturation = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        if not successors[u]:
            res_saturation[u] = res_ratio[u]
        else:
            # Find the dominant resource for u
            dom_res = -1
            max_ratio = -1
            for j in range(n_res):
                if capacities[j] > 0 and demands[u][j] > 0:
                    r = demands[u][j] / capacities[j]
                    if r > max_ratio:
                        max_ratio = r
                        dom_res = j
            # Count successors that also heavily use this resource
            shared_count = 0
            total_succ_weight = 0.0
            for v in successors[u]:
                if dom_res >= 0 and capacities[dom_res] > 0 and demands[v][dom_res] > 0:
                    shared_count += 1
                total_succ_weight += 1.0
            if total_succ_weight > 0:
                res_saturation[u] = res_ratio[u] * (1.0 + 0.5 * shared_count / total_succ_weight)
            else:
                res_saturation[u] = res_ratio[u]

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl_norm.copy()

    # Use smooth-min for slack: instead of raw LST-EST, use a soft version
    # that is less sensitive to ties and provides smoother gradients.
    # soft_slack = -logsumexp(-slack/tau) * tau  (approximates min but smooth)
    tau_slack = 0.5
    neg_slack = -slack
    m_neg = neg_slack.max()
    if m_neg > 1e-10:
        soft_slack = tau_slack * np.log(np.sum(np.exp((neg_slack - m_neg) / tau_slack))) + m_neg
    else:
        soft_slack = np.zeros_like(slack)
    # soft_slack is a smooth approximation of min(slack); lower = less flexible
    soft_slack_norm = normalize(soft_slack)

    # Normalize the new resource saturation signal
    sat_norm = normalize(res_saturation)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return cpl_norm.copy()

    # Combine scores with rebalanced weights:
    # Primary: Critical Path Length (dominant, ~55% of signal)
    # Secondary: Resource contention with multiplicative CPL coupling (~15%)
    # Tertiary: Resource-weighted tail (chains of resource-heavy work, ~15%)
    # Quaternary: Resource saturation depth (congestion chains, ~10%)
    # Quinary: Low soft-slack (less flexible activities first, ~5%)
    # Senary: Duration (tiny tie-break, ~2%)
    scores = (1.0 * cpl_norm * (1.0 + 0.12 * res_norm)
              + 0.18 * tail_norm
              + 0.10 * sat_norm
              - 0.08 * soft_slack_norm
              + 0.03 * dur_norm)

    return scores
