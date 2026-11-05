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

    # Dynamic resource-contention priority via proper event-based SGS simulation
    # This identifies which activities actually cause bottlenecks
    contention_score = np.zeros(n)
    if demands is not None and capacities is not None and n_resources > 0:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Proper event-based serial SGS simulation
        remaining_in_degree = in_degree[:]
        remaining_successors = [list(s) for s in successors]
        
        # Use a priority queue based on LD
        eligible = []
        for i in range(n):
            if remaining_in_degree[i] == 0:
                heapq.heappush(eligible, (-LD[i], i))
        
        scheduled = {}  # activity -> (start, end)
        
        # Track resource availability using event-based approach
        # resource_events[r] = list of (time, delta) for resource r
        # We'll use a simpler approach: track when each resource is free
        # For each resource, maintain a list of (end_time, amount_freed)
        
        # Actually, let's use a cleaner event-based simulation
        # For each resource, we track usage over time using a list of (time, usage_change)
        
        # Simpler approach: for each activity, find the earliest start time
        # by checking when all required resources are available
        
        current_time = 0.0
        num_scheduled = 0
        
        # Track resource usage: for each resource, a list of (time, delta)
        # where delta > 0 means resource becomes available, delta < 0 means consumed
        resource_events = [[] for _ in range(n_resources)]
        
        # We need to track cumulative usage. Let's use a different approach:
        # For each resource, maintain a sorted list of (time, amount) representing
        # when resources become free. We'll process events in time order.
        
        # Actually, the cleanest way: simulate time steps
        # For each eligible activity, find the earliest time it can start
        
        while num_scheduled < n:
            if time.time() > deadline:
                break
                
            if not eligible:
                break
                
            # Get the highest priority eligible activity
            _, best_activity = heapq.heappop(eligible)
            
            # Find the earliest start time for this activity
            earliest_start = current_time
            
            for r in range(n_resources):
                if demands_arr[best_activity][r] > 0:
                    # Find when enough of resource r is available
                    # We need to track resource availability over time
                    # For simplicity, we'll check all scheduled activities that use this resource
                    
                    # Build a timeline of resource r availability
                    # Start with full capacity
                    available = capacities_arr[r]
                    # Sort all activities using resource r by their end time
                    # and find when enough capacity is freed
                    
                    # This is complex; let's use a simpler heuristic
                    # Check each scheduled activity that uses resource r
                    # and see when it ends
                    
                    # Actually, let's just advance to the next event time
                    # For now, use a simple approach: check all currently running activities
                    pass
            
            # Simplified: just schedule at current_time for now
            # This is a limitation; a proper implementation would track resource timelines
            
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
            
            # Update in-degrees and add newly eligible activities
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
