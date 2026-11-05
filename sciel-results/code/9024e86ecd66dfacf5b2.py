import numpy as np
import time
from collections import deque

# Global deadline guard (wall-clock, seconds)
_DEADLINE_START = time.time()
_DEADLINE_LIMIT = 150.0
_SAFETY_MARGIN = 10.0  # stop this much before the hard limit

def _time_budget_left():
    return _DEADLINE_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)

def priority(instance):
    global _DEADLINE_START
    _DEADLINE_START = time.time()  # reset per call for safety

    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance["demands"]
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    R = len(capacities)

    # Build successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # Topological order (Kahn)
    in_deg = [len(preds) for preds in predecessors]
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)

    # LPE (Longest Path to End) in reverse topological order
    lpe = [0.0] * n
    for u in reversed(topo):
        if not successors[u]:
            lpe[u] = durations[u]
        else:
            max_succ_lpe = max(lpe[v] for v in successors[u])
            lpe[u] = durations[u] + max_succ_lpe

    # Demand ratios per resource
    max_ratios = []
    for i in range(n):
        max_r = 0.0
        for r in range(R):
            if capacities[r] > 0:
                ratio = float(demands[i][r]) / float(capacities[r])
                if ratio > max_r:
                    max_r = ratio
        max_ratios.append(max_r)

    # Static base score
    base_scores = np.array([lpe[i] * (1.0 + max_ratios[i]) for i in range(n)], dtype=float)

    # ---- Simulation-based bottleneck detection ----
    # Run a greedy serial schedule using base_scores as priority.
    # Track, for each activity, the total "delay" caused by resource conflicts.
    # This identifies which activities actually contend for resources.
    bottleneck = np.zeros(n, dtype=float)

    if _time_budget_left() > 2.0:
        # Greedy schedule simulation
        remaining = set(range(n))
        # Track resource usage over time: for each resource, list of (end_time, demand)
        # We'll use a simple event-based approach.
        # resource_busy[r] = list of (end_time, demand)
        resource_busy = [[] for _ in range(R)]
        # For each resource, we need to find the earliest time all required capacity is free.
        # Simpler: for each resource, maintain a sorted list of intervals where it's busy.
        # Since n=30, we can just compute the earliest feasible start by checking all intervals.

        # Actually, for a renewable resource, we need to find the earliest time t such that
        # for each resource r, the total demand of activities running at t plus demands[i][r] <= capacities[r].
        # With n=30, we can compute this by checking candidate start times.

        # Let's use a simpler approach: for each activity, compute the earliest start time
        # considering both precedence and resource availability.
        # We'll maintain a set of "scheduled" activities and their start/end times.

        sched_start = [0.0] * n
        sched_end = [0.0] * n
        scheduled = [False] * n

        # We need to find the earliest time at which a given activity can start,
        # given the already-scheduled activities.
        # For renewable resources, we need to check resource availability at each candidate time.

        # Candidate times: 0, and all end times of already-scheduled activities.
        # For each candidate time, check if the activity's resource demands fit.

        # Let's implement a helper: earliest_start(i, candidate_times, resource_busy_intervals)
        # where resource_busy_intervals[r] is a list of (start, end, demand) for resource r.

        # Actually, let's just simulate step by step.
        # At each step, pick the highest-priority eligible activity and schedule it at the earliest feasible time.

        # For resource feasibility at time t for activity i:
        # For each resource r, sum of demands of all scheduled activities that overlap with [t, t+durations[i]]
        # plus demands[i][r] must be <= capacities[r].

        # Since n=30, this is O(n^2 * R) per activity, which is fine.

        def earliest_feasible_start(i, sched_data):
            # sched_data: list of (j, start_j, end_j) for scheduled activities
            # Returns earliest time t >= 0 such that activity i can start at t.
            # Candidate times: 0 and all start_j, end_j for scheduled j.
            candidates = [0.0]
            for (j, sj, ej) in sched_data:
                candidates.append(sj)
                candidates.append(ej)
            candidates = sorted(set(candidates))
            for t in candidates:
                feasible = True
                for r in range(R):
                    if demands[i][r] == 0:
                        continue
                    total = float(demands[i][r])
                    for (j, sj, ej) in sched_data:
                        # Check if [sj, ej] overlaps with [t, t + durations[i]]
                        if sj < t + durations[i] and ej > t:
                            total += float(demands[j][r])
                    if total > capacities[r] + 1e-9:
                        feasible = False
                        break
                if feasible:
                    return t
            # If no candidate works (shouldn't happen), return max candidate
            return candidates[-1] if candidates else 0.0

        sched_data = []
        eligible = [i for i in range(n) if len(predecessors[i]) == 0]
        in_deg_remaining = [len(preds) for preds in predecessors]
        pred_count = [len(preds) for preds in predecessors]

        while len(sched_data) < n:
            # Find eligible activities (all predecessors scheduled)
            eligible = [j for j in range(n) if not scheduled[j] and in_deg_remaining[j] == 0]
            if not eligible:
                break  # Should not happen in a valid DAG
            # Pick highest priority (lowest base_score means higher priority? 
            # Wait: the task says "highest-priority" activity. 
            # In the original, higher score = higher priority.
            # So pick max base_score.
            best = max(eligible, key=lambda j: base_scores[j])
            t = earliest_feasible_start(best, sched_data)
            # Compute bottleneck: delay due to resources
            # The "natural" start would be max end time of predecessors (or 0).
            pred_end = 0.0
            for p in predecessors[best]:
                if sched_end[p] > pred_end:
                    pred_end = sched_end[p]
            delay = t - pred_end
            if delay > 1e-9:
                bottleneck[best] += delay * (1.0 + max_ratios[best])
            sched_start[best] = t
            sched_end[best] = t + durations[best]
            scheduled[best] = True
            sched_data.append((best, t, t + durations[best]))
            # Update in-degrees of successors
            for v in successors[best]:
                in_deg_remaining[v] -= 1

    # Normalize bottleneck scores
    if bottleneck.max() > 1e-9:
        bottleneck_norm = bottleneck / bottleneck.max()
    else:
        bottleneck_norm = bottleneck

    # Blend: score = base + weight * bottleneck_norm
    # We want the bottleneck signal to be a moderate adjustment, not dominate.
    # base_scores are on the scale of LPE * (1+ratio), which is ~durations.
    # bottleneck_norm is in [0,1]. We need to scale it to be comparable.
    # Let's scale by the mean of base_scores.
    base_mean = base_scores.mean() if n > 0 else 1.0
    weight = 0.5 * base_mean  # moderate influence
    final_scores = base_scores + weight * bottleneck_norm

    return final_scores
