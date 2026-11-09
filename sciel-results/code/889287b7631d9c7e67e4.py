# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np

_DEADLINE_SECONDS = 140.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _simulate_earliest_start(durations, demands, capacities, predecessors, n):
    """Simulate a simple earliest-start schedule to compute temporal resource pressure."""
    # Compute earliest start times ignoring resources (ES)
    es = np.zeros(n, dtype=float)
    # Topological order
    in_deg = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)
                in_deg[j] += 1

    top_q = [j for j in range(n) if in_deg[j] == 0]
    topo = []
    while top_q:
        u = top_q.pop(0)
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                top_q.append(v)

    for u in topo:
        for v in successors[u]:
            if es[u] + durations[u] > es[v]:
                es[v] = es[u] + durations[u]

    # Now simulate serial SGS with all activities having same priority (arbitrary order = topo)
    # Track resource usage over time
    # For each activity, record its [start, end) interval and contribution to peak
    start_times = np.zeros(n, dtype=float)
    end_times = np.zeros(n, dtype=float)

    # We'll do a simple serial SGS: iterate through topological order,
    # for each activity, find earliest feasible start given resource constraints
    # This is O(n^2 * n_resources) which is fine for n=30

    # Resource usage: for each resource k, track usage over time
    # Use a list of (time, delta) events per resource
    # Simpler: for each activity j, try to start at es[j], then incrementally delay if resource conflict

    # Pre-compute for efficiency: for each activity j, its demand vector
    # For checking if activity j can start at time t: for each resource k,
    #   sum of demands[i][k] for all i with start_times[i] < t < end_times[i] + demands[j][k] <= capacities[k]

    active = [False] * n
    for j in topo:
        t = es[j]
        # Check if resources are available at t; if not, delay
        while True:
            feasible = True
            for k in range(len(capacities)):
                if capacities[k] <= 0:
                    continue
                # Compute current usage of resource k at time t
                usage = 0.0
                for i in range(n):
                    if active[i]:
                        if start_times[i] < t < end_times[i]:
                            usage += demands[i][k]
                if usage + demands[j][k] > capacities[k] + 1e-9:
                    feasible = False
                    # Find earliest time when some active activity on resource k ends
                    earliest_end = float('inf')
                    for i in range(n):
                        if active[i] and start_times[i] < t < end_times[i]:
                            if end_times[i] < earliest_end:
                                earliest_end = end_times[i]
                    if earliest_end < float('inf'):
                        t = earliest_end
                    else:
                        t += 1.0  # safety
                    break
            if feasible:
                break
            # Safety: if we've been looping too long
            if t > 10000:
                break

        start_times[j] = t
        end_times[j] = t + durations[j]
        active[j] = True

    # Now compute for each activity j, the peak resource utilization it contributes to
    # For each resource k, find the maximum concurrent demand that includes j
    # This is expensive; instead, compute the "bottleneck pressure" as:
    # For each activity j and resource k: (demands[j][k] / capacities[k]) * (duration_j / max_duration)
    # weighted by how much it overlaps with others

    # Simpler proxy: for each activity j, compute the average resource utilization
    # during its execution window across all resources
    resource_pressure = np.zeros(n, dtype=float)
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0

    for j in range(n):
        s = start_times[j]
        e = end_times[j]
        dur = durations[j]
        if dur <= 0:
            continue
        # Sample at midpoint
        mid = (s + e) / 2.0
        for k in range(len(capacities)):
            if capacities[k] <= 0:
                continue
            usage = 0.0
            for i in range(n):
                if start_times[i] < mid < end_times[i]:
                    usage += demands[i][k]
            # Pressure: how much of capacity is used, and how much j contributes
            frac = usage / capacities[k]
            j_contrib = demands[j][k] / max(usage, 1e-9)
            resource_pressure[j] += frac * j_contrib

    return start_times, end_times, resource_pressure


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    if n == 0:
        return np.array([])

    # Build successors
    successors = [[] for _ in range(n)]
    for j in range(n):
        if predecessors[j]:
            for p in predecessors[j]:
                successors[p].append(j)

    # Calculate in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        if predecessors[j]:
            in_degree[j] = len(predecessors[j])

    # Topological sort (Kahn's algorithm)
    queue = []
    for j in range(n):
        if in_degree[j] == 0:
            queue.append(j)

    top_order = []
    while queue:
        u = queue.pop(0)
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Reverse topological order
    rev_top_order = list(reversed(top_order))

    # Calculate longest path from each node to the end (critical path length)
    longest_path_to_end = np.zeros(n, dtype=float)
    for i in rev_top_order:
        max_succ = 0.0
        for j in successors[i]:
            if durations[j] + longest_path_to_end[j] > max_succ:
                max_succ = durations[j] + longest_path_to_end[j]
        longest_path_to_end[i] = durations[i] + max_succ

    # Deadline guard: if we've used too much time, fall back to simple heuristic
    if _remaining_budget() < 5.0:
        # Simple fallback: critical path + basic resource ratio
        max_ratio = np.zeros(n, dtype=float)
        for j in range(n):
            mx = 0.0
            for k in range(len(capacities)):
                if capacities[k] > 0:
                    r = demands[j][k] / capacities[k]
                    if r > mx:
                        mx = r
            max_ratio[j] = mx
        max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
        if max_cp == 0:
            max_cp = 1.0
        norm_cp = longest_path_to_end / max_cp
        max_mr = np.max(max_ratio) if n > 0 else 1.0
        if max_mr == 0:
            max_mr = 1.0
        norm_max_ratio = max_ratio / max_mr
        return norm_cp * (1.0 + 0.6 * norm_max_ratio)

    # Compute temporal resource pressure via simulation
    start_times, end_times, resource_pressure = _simulate_earliest_start(
        durations, demands, capacities, predecessors, n
    )

    # Also compute simple max ratio as a baseline
    max_ratio = np.zeros(n, dtype=float)
    for j in range(n):
        mx = 0.0
        for k in range(len(capacities)):
            if capacities[k] > 0:
                r = demands[j][k] / capacities[k]
                if r > mx:
                    mx = r
        max_ratio[j] = mx

    # Normalize components
    max_cp = np.max(longest_path_to_end) if n > 0 else 1.0
    if max_cp == 0:
        max_cp = 1.0
    norm_cp = longest_path_to_end / max_cp

    max_mr = np.max(max_ratio) if n > 0 else 1.0
    if max_mr == 0:
        max_mr = 1.0
    norm_max_ratio = max_ratio / max_mr

    max_rp = np.max(resource_pressure) if n > 0 else 1.0
    if max_rp == 0:
        max_rp = 1.0
    norm_rp = resource_pressure / max_rp

    # Combine: critical path tail + resource pressure (temporal) + simple ratio
    # Weight temporal pressure more heavily as it captures actual contention
    score = norm_cp * (1.0 + 0.4 * norm_max_ratio + 0.6 * norm_rp)

    return score
# EVOLVE-BLOCK-END
