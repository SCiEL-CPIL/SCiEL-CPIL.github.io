"""Resource-Constrained Project Scheduling: priority scoring via serial schedule generation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def _build_graph(instance):
    """Precompute adjacency, durations, demands, capacities, predecessors."""
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)

    # Build successor map and in-degree
    succ = defaultdict(list)
    in_deg = np.zeros(n, dtype=int)
    if predecessors is not None:
        for i, preds in enumerate(predecessors):
            for p in preds:
                succ[p].append(i)
                in_deg[i] += 1

    return n, durations, demands, capacities, predecessors, succ, in_deg


def _serial_schedule(n, durations, demands, capacities, predecessors, succ, in_deg,
                     priority_scores, time_limit=None):
    """
    Serial schedule generation: repeatedly pick the highest-priority eligible activity
    that fits within remaining resource capacities, schedule it as early as possible.
    Returns makespan.
    """
    if time_limit is not None:
        if time.time() - time_limit[0] > time_limit[1]:
            return None

    # remaining in-degree (copy)
    rem_in_deg = in_deg.copy()
    # scheduled set
    scheduled = [False] * n
    # current time per activity (start time)
    start_times = np.full(n, -1.0)
    # resource usage over time: we track events
    # For simplicity with 30 activities, we can simulate time in steps
    # Actually, let's do event-based simulation

    # Track resource availability: list of (end_time, resource_idx, amount)
    resource_events = []  # (end_time, res_idx, amount)
    current_time = 0.0

    # Eligible: in_deg == 0 and not scheduled
    for step in range(n):
        if time_limit is not None and time.time() - time_limit[0] > time_limit[1]:
            return None

        # Find eligible activities
        eligible = []
        for i in range(n):
            if not scheduled[i] and rem_in_deg[i] == 0:
                eligible.append(i)

        if not eligible:
            # Should not happen if precedence is valid
            break

        # Pick highest priority eligible activity
        best_act = max(eligible, key=lambda i: priority_scores[i])

        # Determine earliest start time for best_act
        # Must wait for all predecessors to finish
        est = 0.0
        if predecessors is not None:
            for p in predecessors[best_act]:
                if start_times[p] >= 0:
                    est = max(est, start_times[p] + durations[p])

        # Check resource availability: find earliest time >= est where all resources are available
        # We need to find a time t >= est such that for each resource r,
        # the sum of demands of activities running at t + demand[best_act][r] <= capacity[r]

        # Collect all event times >= est
        event_times = set([est])
        for (end_t, _, _) in resource_events:
            if end_t >= est:
                event_times.add(end_t)

        # Sort candidate start times
        candidate_starts = sorted(event_times)

        # Also need to consider: at any time between events, resource usage is constant
        # So we just need to check at each candidate_start
        chosen_start = None
        for cs in candidate_starts:
            if cs < est:
                continue
            # Check if best_act can start at cs
            can_start = True
            for r in range(len(capacities) if capacities else 0):
                # Sum demands of activities running at time cs
                usage = 0.0
                if demands is not None:
                    for j in range(n):
                        if scheduled[j] and start_times[j] >= 0:
                            if start_times[j] <= cs < start_times[j] + durations[j]:
                                usage += demands[j][r]
                if usage + (demands[best_act][r] if demands is not None else 0) > capacities[r]:
                    can_start = False
                    break
            if can_start:
                chosen_start = cs
                break

        if chosen_start is None:
            # Fallback: try later times
            # This shouldn't happen if we have enough event points, but let's be safe
            # Try a grid
            t = est
            while True:
                can_start = True
                for r in range(len(capacities) if capacities else 0):
                    usage = 0.0
                    if demands is not None:
                        for j in range(n):
                            if scheduled[j] and start_times[j] >= 0:
                                if start_times[j] <= t < start_times[j] + durations[j]:
                                    usage += demands[j][r]
                    if usage + (demands[best_act][r] if demands is not None else 0) > capacities[r]:
                        can_start = False
                        break
                if can_start:
                    chosen_start = t
                    break
                t += 1.0
                if t > 10000:
                    # Give up, just use est
                    chosen_start = est
                    break

        # Schedule best_act
        scheduled[best_act] = True
        start_times[best_act] = chosen_start
        end_time = chosen_start + durations[best_act]

        # Add resource events
        if demands is not None:
            for r in range(len(capacities) if capacities else 0):
                resource_events.append((end_time, r, demands[best_act][r]))

        # Update in-degrees of successors
        for s in succ[best_act]:
            rem_in_deg[s] -= 1

    # Compute makespan
    makespan = 0.0
    for i in range(n):
        if scheduled[i]:
            makespan = max(makespan, start_times[i] + durations[i])
    return makespan


def _compute_priority_scores(instance):
    """
    Compute priority scores using a combination of:
    1. Critical path length (longest path from activity to end)
    2. Resource contention (how much this activity shares resources with successors)
    3. Duration (longer activities first)
    """
    n, durations, demands, capacities, predecessors, succ, in_deg = _build_graph(instance)

    # Compute longest path from each node to a sink (topological order)
    # Use reverse topological order
    # First, get topological order
    from collections import deque
    in_deg_copy = in_deg.copy()
    queue = deque([i for i in range(n) if in_deg_copy[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    # Longest path from node to end (reverse topo)
    longest_to_end = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        longest_to_end[u] = durations[u]
        for v in succ[u]:
            longest_to_end[u] = max(longest_to_end[u], durations[u] + longest_to_end[v])

    # Resource contention: for each activity, sum over successors of shared resource demand
    # Metric: how much this activity's resource usage blocks successors
    contention = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        n_res = len(capacities)
        for u in range(n):
            for v in succ[u]:
                # Shared resource pressure: sum over resources of min(demand_u[r], demand_v[r]) / capacity[r]
                for r in range(n_res):
                    shared = min(demands[u][r], demands[v][r])
                    if capacities[r] > 0:
                        contention[u] += shared / capacities[r]

    # Combine: weighted sum
    # Normalize each component
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    score_cp = normalize(longest_to_end)
    score_dur = normalize(durations)
    score_cont = normalize(contention)

    # Weights: critical path is most important, then contention, then duration
    scores = 0.5 * score_cp + 0.35 * score_cont + 0.15 * score_dur

    return scores


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority.
    """
    start_time = time.time()
    time_limit = [start_time, 100.0]  # Leave 50s safety margin

    try:
        scores = _compute_priority_scores(instance)
    except Exception:
        # Fallback: duration-based
        n = len(instance["durations"])
        durations = np.asarray(instance["durations"], dtype=float)
        return durations

    # Optionally, refine scores by doing a few serial schedules with perturbed priorities
    # to estimate which activities lead to shorter makespans
    n, durations, demands, capacities, predecessors, succ, in_deg = _build_graph(instance)

    # Try a few perturbations to see if we can improve
    best_scores = scores.copy()
    best_makespan = None

    # Baseline makespan
    baseline_ms = _serial_schedule(n, durations, demands, capacities, predecessors, succ, in_deg,
                                   scores, time_limit)
    if baseline_ms is not None:
        best_makespan = baseline_ms

    # Try a few random perturbations
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() - time_limit[0] > time_limit[1] - 10:
            break
        # Perturb scores
        perturbed = scores + rng.normal(0, 0.1, n)
        ms = _serial_schedule(n, durations, demands, capacities, predecessors, succ, in_deg,
                              perturbed, time_limit)
        if ms is not None and (best_makespan is None or ms < best_makespan):
            best_makespan = ms
            best_scores = perturbed

    return best_scores
# EVOLVE-BLOCK-END
