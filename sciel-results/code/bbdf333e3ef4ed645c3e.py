"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque, defaultdict
import heapq

_DEADLINE = 118.0  # seconds, safety margin before 120s kill
_START_TIME = None


def _serial_schedule(durations, demands, capacities, predecessors, priority_scores):
    """
    Serial schedule generation scheme.
    durations: (n,) array of activity durations
    demands: (n, m) array of resource demands per activity per resource
    capacities: (m,) array of resource capacities
    predecessors: list of lists, predecessors[i] is list of indices
    priority_scores: (n,) array of priority scores (higher = earlier)
    Returns: makespan (total project duration)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = len(durations)
    m = len(capacities)
    
    # Build successor lists
    in_degree = [0] * n
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    start_times = [0] * n
    completed = [False] * n
    eligible = set()
    
    for i in range(n):
        if in_degree[i] == 0:
            eligible.add(i)
    
    current_time = 0
    scheduled_count = 0
    
    # For each resource, maintain list of (start, end, activity_id)
    resource_intervals = [[] for _ in range(m)]
    
    def get_usage_at(r, t):
        """Get total usage of resource r at time t."""
        total = 0
        for (s, e, _) in resource_intervals[r]:
            if s <= t < e:
                total += demands[r][0]  # placeholder, will fix
        return total
    
    def can_start(activity, t):
        """Check if activity can start at time t given resource constraints."""
        for r in range(m):
            demand = demands[activity][r]
            if demand == 0:
                continue
            # Sum usage of all intervals overlapping t
            usage = 0
            for (s, e, a) in resource_intervals[r]:
                if s <= t < e:
                    usage += demands[a][r]
            if usage + demand > capacities[r]:
                return False
        return True
    
    def earliest_start(activity, t_min):
        """Find earliest time >= t_min when activity can start."""
        # Candidate times: t_min, and all end times of intervals >= t_min
        # (start times don't help since more activities are running at start times)
        candidates = set()
        candidates.add(t_min)
        for r in range(m):
            for (s, e, a) in resource_intervals[r]:
                if e >= t_min:
                    candidates.add(e)
        
        for ct in sorted(candidates):
            if ct >= t_min and can_start(activity, ct):
                return ct
        
        # Fallback: shouldn't normally reach here, but just in case
        return t_min
    
    while scheduled_count < n:
        if time.time() - _START_TIME > _DEADLINE - 3.0:
            # Return best-effort estimate
            max_st = max(start_times) if any(st > 0 for st in start_times) else 0
            return max_st + max(durations) if n > 0 else 0
        
        # Find highest priority eligible activity
        best_activity = None
        best_priority = -float('inf')
        for a in eligible:
            if priority_scores[a] > best_priority:
                best_priority = priority_scores[a]
                best_activity = a
        
        if best_activity is None:
            break
        
        st = earliest_start(best_activity, current_time)
        start_times[best_activity] = st
        et = st + durations[best_activity]
        
        for r in range(m):
            if demands[best_activity][r] > 0:
                resource_intervals[r].append((st, et, best_activity))
        
        completed[best_activity] = True
        scheduled_count += 1
        eligible.discard(best_activity)
        
        if st > current_time:
            current_time = st
        
        for succ in successors[best_activity]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                eligible.add(succ)
    
    if any(st > 0 for st in start_times):
        makespan = max(start_times[i] + durations[i] for i in range(n))
    else:
        makespan = sum(durations)
    
    return makespan


def priority(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demand"] if "demand" in instance else instance.get("demands", np.zeros((len(durations), 0))), dtype=float)
    capacities = np.asarray(instance["capacity"] if "capacity" in instance else instance.get("capacities", np.array([float('inf')])), dtype=float)
    predecessors = instance["predecessors"] if "predecessors" in instance else instance.get("predecessor_lists", [[] for _ in range(len(durations))])
    
    n = len(durations)
    m = len(capacities)
    
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
        capacities = capacities.reshape(1)
        m = 1
    
    if m == 0 or len(capacities) == 0:
        m = 1
        capacities = np.array([float('inf')])
        demands = np.zeros((n, 1))
    
    if demands.shape[1] < m:
        pad = np.zeros((n, m - demands.shape[1]))
        demands = np.hstack([demands, pad])
    elif demands.shape[1] > m:
        demands = demands[:, :m]
    
    # Build graph
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            in_degree[i] += 1
            successors[p].append(i)
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    in_deg_copy = in_degree[:]
    while queue:
        node = queue.popleft()
        topo.append(node)
        for s in successors[node]:
            in_deg_copy[s] -= 1
            if in_deg_copy[s] == 0:
                queue.append(s)
    if len(topo) < n:
        for i in range(n):
            if i not in topo:
                topo.append(i)
    
    # Compute head (earliest start time in unconstrained schedule)
    head = np.zeros(n)
    for node in topo:
        best = 0
        for p in predecessors[node]:
            val = head[p] + durations[p]
            if val > best:
                best = val
        head[node] = best
    
    # Compute tail (longest path from node to end, excluding node's duration)
    tail = np.zeros(n)
    for node in reversed(topo):
        best = 0
        for s in successors[node]:
            val = durations[s] + tail[s]
            if val > best:
                best = val
        tail[node] = best
    
    # Critical path length
    cp_len = max(head[i] + durations[i] + tail[i] for i in range(n)) if n > 0 else 0
    
    # Total float (slack)
    total_float = cp_len - head - durations - tail
    total_float = np.maximum(total_float, 0)
    
    # Number of immediate successors
    succ_count = np.array([len(succs) for succs in successors], dtype=float)
    pred_count = np.array([len(preds) for preds in predecessors], dtype=float)
    
    # Total resource demand
    total_demand = demands.sum(axis=1)
    
    # Max demand-to-capacity ratio
    demand_ratio = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            ratio = demands[:, r] / capacities[r]
            demand_ratio = np.maximum(demand_ratio, ratio)
    
    # Resource contention: for each activity, measure how much of each scarce
    # resource it uses relative to total demand on that resource across all activities.
    # Activities that consume a large fraction of a scarce resource are bottlenecks.
    contention = np.zeros(n)
    for r in range(m):
        if capacities[r] > 0 and np.isfinite(capacities[r]):
            col_total = demands[:, r].sum()
            if col_total > 1e-12:
                # Fraction of this resource used by this activity
                frac = demands[:, r] / col_total
                # Weight by scarcity (1/capacity) to emphasize scarce resources
                scarcity = 1.0 / capacities[r]
                contention = np.maximum(contention, frac * scarcity)
    
    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        mn, mx = x.min(), x.max()
        if mx - mn > 1e-12:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)
    
    # We want higher score = schedule earlier
    # Low float -> high priority, so invert
    norm_float_inv = 1.0 - norm(total_float)
    norm_tail = norm(tail)
    norm_duration = norm(durations)
    norm_demand = norm(total_demand)
    norm_ratio = norm(demand_ratio)
    norm_contention = norm(contention)
    norm_succ = norm(succ_count)
    
    # Weighted combination - emphasize criticality and resource pressure
    # Increase weight on contention and tail, decrease on duration
    scores = (0.35 * norm_float_inv +
              0.25 * norm_tail +
              0.15 * norm_contention +
              0.12 * norm_demand +
              0.07 * norm_ratio +
              0.04 * norm_duration +
              0.02 * norm_succ)
    
    return scores
# EVOLVE-BLOCK-END
