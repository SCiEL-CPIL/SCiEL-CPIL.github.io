import numpy as np
from collections import deque
import time
import heapq

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)

    if n == 0:
        return np.array([])

    # Extract resource info
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    n_resources = len(capacities) if capacities is not None else 0

    # Build successors list and in-degrees
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological Sort (Kahn's Algorithm)
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

    if len(topo_order) < n:
        return durations

    # Calculate Longest Path to End (LD)
    LD = np.zeros(n)
    for u in reversed(topo_order):
        max_succ_LD = 0.0
        for v in successors[u]:
            if LD[v] > max_succ_LD:
                max_succ_LD = LD[v]
        LD[u] = durations[u] + max_succ_LD

    # Calculate demand-to-capacity ratio (resource pressure)
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        for i in range(n):
            if i < len(demands_arr):
                for r in range(n_resources):
                    if capacities_arr[r] > 0:
                        ratio = demands_arr[i][r] / capacities_arr[r]
                        if ratio > resource_pressure[i]:
                            resource_pressure[i] = ratio

    # Dynamic resource-contention priority via simulated greedy schedule
    # This identifies which activities actually cause bottlenecks
    contention_score = np.zeros(n)
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Simulate a greedy schedule using LD as priority
        # Track resource usage over time to identify contention
        # Since n=30, we can simulate discretely
        
        # First, create a schedule using LD priority
        # This is a simplified serial SGS simulation
        remaining_in_degree = in_degree[:]  # reset
        remaining_successors = [list(s) for s in successors]
        
        # Use a priority queue based on LD
        eligible = []
        for i in range(n):
            if remaining_in_degree[i] == 0:
                heapq.heappush(eligible, (-LD[i], i))
        
        scheduled = {}  # activity -> (start, end)
        resource_usage = [0.0] * n_resources  # current usage at each time step
        
        # We need to track resource availability over time
        # For simplicity, track events (time, delta_resource)
        events = {}  # time -> list of (resource, delta)
        
        current_time = 0.0
        num_scheduled = 0
        
        while eligible and num_scheduled < n:
            if time.time() > deadline:
                break
                
            # Find earliest time when resource becomes available for top eligible
            best_time = current_time
            best_activity = None
            best_key = None
            
            # Get all eligible activities and find the one that can start earliest
            candidates = list(eligible)
            eligible.clear()
            
            for key, act in sorted(candidates):
                # Check if this activity can start at current_time
                can_start = True
                for r in range(n_resources):
                    if demands_arr[act][r] > capacities_arr[r]:
                        can_start = False
                        break
                
                if can_start:
                    best_activity = act
                    best_key = key
                    break
                else:
                    # Need to wait for resources; approximate by checking
                    # when resources free up (simplified: just use current_time)
                    pass
            
            if best_activity is None:
                # No activity can start; advance time (simplified)
                # In practice, we'd need event-based simulation
                # For now, just schedule in LD order ignoring resources
                # This is a fallback
                for key, act in sorted(candidates):
                    scheduled[act] = (current_time, current_time + durations[act])
                    num_scheduled += 1
                    # Update in-degrees
                    for v in remaining_successors[act]:
                        remaining_in_degree[v] -= 1
                        if remaining_in_degree[v] == 0:
                            heapq.heappush(eligible, (-LD[v], v))
                    current_time += durations[act]
                break
            
            # Schedule the best activity
            heapq.heappush(eligible, (best_key, best_activity))
            heapq.heappop(eligible)
            
            start = current_time
            end = start + durations[best_activity]
            scheduled[best_activity] = (start, end)
            
            # Calculate contention: how much this activity competes
            # with other unscheduled activities for resources
            contention = 0.0
            for r in range(n_resources):
                if demands_arr[best_activity][r] > 0 and capacities_arr[r] > 0:
                    # Count how many other activities also need this resource
                    competitors = 0
                    for j in range(n):
                        if j != best_activity and j not in scheduled:
                            if demands_arr[j][r] > 0:
                                competitors += 1
                    # Higher contention if many competitors and high demand ratio
                    ratio = demands_arr[best_activity][r] / capacities_arr[r]
                    contention += ratio * competitors
            
            contention_score[best_activity] = contention
            
            # Update in-degrees
            for v in remaining_successors[best_activity]:
                remaining_in_degree[v] -= 1
                if remaining_in_degree[v] == 0:
                    heapq.heappush(eligible, (-LD[v], v))
            
            num_scheduled += 1
            current_time = end

    # Normalize all components
    LD_min, LD_max = LD.min(), LD.max()
    LD_range = LD_max - LD_min if LD_max > LD_min else 1.0
    LD_norm = (LD - LD_min) / LD_range

    rp_min, rp_max = resource_pressure.min(), resource_pressure.max()
    rp_range = rp_max - rp_min if rp_max > rp_min else 1.0
    rp_norm = (resource_pressure - rp_min) / rp_range

    cs_min, cs_max = contention_score.min(), contention_score.max()
    cs_range = cs_max - cs_min if cs_max > cs_min else 1.0
    cs_norm = (contention_score - cs_min) / cs_range

    # Weighted combination:
    # - Critical path (LD): 55%
    # - Own resource pressure: 15%
    # - Dynamic contention from simulated schedule: 30%
    combined_score = 0.55 * LD_norm + 0.15 * rp_norm + 0.30 * cs_norm

    # Add tiny tie-breaker for numerical stability
    return combined_score + durations * 1e-9
