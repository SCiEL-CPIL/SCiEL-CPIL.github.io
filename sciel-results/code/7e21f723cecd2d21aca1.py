# EVOLVE-BLOCK-START
import time
import heapq
import numpy as np
import random

# Explicit wall-clock deadline guard (evaluator kills at 150s).
_DEADLINE_SECONDS = 145.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)


def _serial_sgs_fast(durations, demands, capacities, predecessors, priority_vec):
    """
    Fast serial SGS. Returns (makespan, order).
    Assumes predecessors is a list of lists.
    """
    n = len(durations)
    remaining_preds = np.zeros(n, dtype=int)
    for j in range(n):
        p = predecessors[j]
        if p is not None:
            remaining_preds[j] = len(p)
    
    # Successor map
    successors = [[] for _ in range(n)]
    for j in range(n):
        p = predecessors[j]
        if p is not None:
            for i in p:
                successors[i].append(j)

    eligible = []
    for j in range(n):
        if remaining_preds[j] == 0:
            heapq.heappush(eligible, (-priority_vec[j], j))

    order = []
    resource_ends = [[] for _ in range(len(capacities))]
    n_scheduled = 0
    makespan = 0.0

    while n_scheduled < n:
        if not eligible:
            # Should not happen if predecessors are valid, but break to avoid infinite loop
            break
            
        neg_p, j = heapq.heappop(eligible)
        dur = durations[j]
        
        # Find earliest feasible start
        candidates = {0.0}
        for r in range(len(capacities)):
            if demands[j][r] <= 0:
                continue
            for (end, amt) in resource_ends[r]:
                candidates.add(float(end))
        
        cand_list = sorted(candidates)
        feasible_start = cand_list[-1] if cand_list else 0.0
        
        for t in cand_list:
            ok = True
            for r in range(len(capacities)):
                d = demands[j][r]
                if d <= 0:
                    continue
                # Check if resource r is sufficient at time t for duration dur
                # Calculate usage in [t, t+dur]
                # Optimization: Only check if demand > 0
                # We need to ensure that for all tau in [t, t+dur], usage(tau) + d <= cap[r]
                # Usage is piecewise constant, changing only at end times of other activities.
                # So we check at t and at all end times e such that t < e <= t+dur.
                
                # Get all end times in (t, t+dur]
                # resource_ends[r] is sorted by end time
                # We can iterate through them
                max_usage = -1.0
                # Base usage at time t
                base = 0.0
                for (end, amt) in resource_ends[r]:
                    if end > t:
                        base += amt
                
                cur = base
                # Check at t
                if cur > max_usage:
                    max_usage = cur
                    
                # Check at each end time in (t, t+dur]
                for (end, amt) in resource_ends[r]:
                    if t < end <= t + dur:
                        if cur > max_usage:
                            max_usage = cur
                        cur -= amt
                    elif end > t + dur:
                        break
                    # If end <= t, we already counted in base? 
                    # Wait, base counts end > t. So if end <= t, it's not in base.
                    # But if end <= t, the activity has finished, so it doesn't contribute to usage in [t, ...].
                    # So base is correct.
                    
                if max_usage + d > capacities[r] + 1e-9:
                    ok = False
                    break
            if ok:
                feasible_start = t
                break
        
        end = feasible_start + dur
        makespan = max(makespan, end)
        
        for r in range(len(capacities)):
            if demands[j][r] > 0:
                resource_ends[r].append((end, float(demands[j][r])))
                # Keep sorted for efficiency in next iterations? 
                # n is small (30), so sorting is cheap.
                resource_ends[r].sort()
        
        order.append(j)
        n_scheduled += 1
        
        for k in successors[j]:
            remaining_preds[k] -= 1
            if remaining_preds[k] == 0:
                heapq.heappush(eligible, (-priority_vec[k], k))
    
    return makespan, order


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Ensure predecessors is a list of lists for consistent access
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    else:
        # Normalize None to []
        norm_preds = []
        for p in predecessors:
            if p is None:
                norm_preds.append([])
            else:
                norm_preds.append(p)
        predecessors = norm_preds

    # 1. Generate an initial priority vector using a robust heuristic
    # Combine Critical Path (backward) and Resource Demand
    # Backward Pass for CPM
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
    
    # Topological sort
    indeg = np.zeros(n, dtype=int)
    for j in range(n):
        indeg[j] = len(predecessors[j])
    
    from collections import deque
    q = deque([j for j in range(n) if indeg[j] == 0])
    topo_order = []
    indeg_copy = indeg.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                q.append(v)
                
    # If topo sort failed (cycle?), fallback to random
    if len(topo_order) < n:
        return np.random.rand(n)

    # Backward longest path
    dist = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        best = 0.0
        for v in successors[u]:
            if dist[v] > best:
                best = dist[v]
        dist[u] = durations[u] + best
        
    # Resource pressure heuristic
    res_pressure = demands.sum(axis=1) / (capacities + 1e-9)
    # Normalize to similar scale
    if res_pressure.max() > 0:
        res_pressure = res_pressure / res_pressure.max()
        
    # Initial priority: Weighted sum
    w1 = 0.6
    w2 = 0.4
    initial_prio = w1 * (dist / (dist.max() + 1e-9)) + w2 * res_pressure
    
    # 2. Local Search / Hill Climbing on the Priority Vector
    # We will perturb the priority vector and check if makespan improves.
    # Since the SGS is deterministic given the priority order (mostly), 
    # we can effectively search over permutations or priority assignments.
    
    best_makespan, best_order = _serial_sgs_fast(durations, demands, capacities, predecessors, initial_prio)
    best_prio = initial_prio.copy()
    
    if _remaining_budget() < 1.0:
        return best_prio

    # Local Search Strategy:
    # Randomly select two activities and swap their priorities, or perturb one.
    # Given the SGS mechanism, swapping priorities of two activities that are "close" in the schedule might help.
    # A simpler robust approach: Random Restart + Hill Climbing on the priority vector.
    
    rng = random.Random(42)
    best_prio = initial_prio.copy()
    
    # Number of iterations based on time
    # Each SGS run is O(n^2 * R) roughly. n=30, R=4. Very fast.
    # We can do thousands of iterations.
    
    current_prio = best_prio.copy()
    current_ms, _ = _serial_sgs_fast(durations, demands, capacities, predecessors, current_prio)
    
    # Simulated Annealing-like or Simple Hill Climbing
    # Let's do Simple Hill Climbing with random restarts if stuck.
    
    while _remaining_budget() > 2.0:
        # Generate a neighbor
        # Option 1: Swap two random priorities
        # Option 2: Perturb one priority
        # Option 3: Re-randomize a subset
        
        # Let's try swapping two activities in the priority vector.
        # This changes the order in which they are considered if they are both eligible at some point.
        i, j = rng.sample(range(n), 2)
        
        neighbor_prio = current_prio.copy()
        # Swap values
        neighbor_prio[i], neighbor_prio[j] = neighbor_prio[j], neighbor_prio[i]
        
        neighbor_ms, _ = _serial_sgs_fast(durations, demands, capacities, predecessors, neighbor_prio)
        
        if neighbor_ms < current_ms - 1e-9:
            current_ms = neighbor_ms
            current_prio = neighbor_prio
            if current_ms < best_makespan - 1e-9:
                best_makespan = current_ms
                best_prio = current_prio.copy()
        # If no improvement for a while, we might want to restart, 
        # but with 120s budget and fast SGS, simple hill climbing on 30 items is likely sufficient 
        # or we can add a restart mechanism.
        
        # Let's add a simple restart if we haven't improved in the last 500 steps (approx)
        # To keep it simple and fast, we'll just keep going. 
        # If stuck, the random swaps will eventually explore.
        
    return best_prio

# EVOLVE-BLOCK-END
