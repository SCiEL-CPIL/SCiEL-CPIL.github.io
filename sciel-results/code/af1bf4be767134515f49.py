"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Resource-Aware Critical Path Priority Heuristic.
    
    Strategy:
    1. Compute Bottom Level (BL) - longest path to project end.
    2. Simulate SGS with BL priority to get reference start times.
    3. For each activity, compute:
       a) Regret: delay imposed on successors (weighted by successor BL)
       b) Resource Scarcity: max(demand/capacity) across resources, weighted by BL
    4. Priority = weighted combination of BL, Regret, and Resource Scarcity
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance.get("predecessors", [[] for _ in range(n)])
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None or capacities is None:
        return _bottom_level(n, durations, predecessors)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    n_resources = len(capacities)
    
    # Build graph structures
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
            in_degree[i] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
                
    if len(topo_order) != n:
        return durations.copy()

    # Static Metrics
    # Bottom Level (Longest path to end)
    bl = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bl[node] = durations[node]
        else:
            bl[node] = durations[node] + max(bl[s] for s in successors[node])
            
    # Resource Scarcity Score: max(demand/capacity) for each activity
    # High scarcity = activity uses a lot of a scarce resource
    resource_scarcity = np.zeros(n)
    for i in range(n):
        max_ratio = 0.0
        for r in range(n_resources):
            if capacities[r] > 0:
                ratio = demands[i][r] / capacities[r]
                if ratio > max_ratio:
                    max_ratio = ratio
        resource_scarcity[i] = max_ratio

    # Simulate SGS to find "Earliest Possible Start" (EPS) for each activity
    # We assume a greedy schedule where we pick the highest BL activity first.
    # This gives us a reference schedule to evaluate "regret".
    scheduled = [False] * n
    start_times = [0.0] * n
    # Resource usage: list of (end_time, demand) for each resource
    resource_usage = [[] for _ in range(n_resources)]
    
    # We need to simulate the actual scheduling process to know the context.
    # However, the priority function is called once. We must return static scores.
    # The "Lookahead" here is: 
    # If I schedule i now, at time T_i, how much do I delay my successors?
    # Since we don't know T_i exactly without running the full SGS, we estimate T_i 
    # using the resource availability at the earliest precedence start time.
    
    # Let's compute a "Resource Constraint Score" for each activity.
    # For each activity i, check if it can start at its precedence EST.
    # If not, how much delay?
    
    est_prec = np.zeros(n)
    for node in topo_order:
        if predecessors[node]:
            est_prec[node] = max(est_prec[p] + durations[p] for p in predecessors[node])
        else:
            est_prec[node] = 0.0

    # Simulate SGS to estimate "actual" start times under BL priority
    sim_starts = [0.0] * n
    sim_usage = [[] for _ in range(n_resources)]
    
    eligible = set(i for i in range(n) if in_degree[i] == 0)
    sim_in_deg = in_degree[:]
    
    while eligible:
        # Pick highest BL from eligible
        current = max(eligible, key=lambda x: bl[x])
        
        # Determine start time
        earliest_prec = 0.0
        if predecessors[current]:
            earliest_prec = max(sim_starts[p] + durations[p] for p in predecessors[current])
            
        # Check resources
        earliest_res = 0.0
        for r in range(n_resources):
            d = demands[current][r]
            if d > 0:
                t = earliest_prec
                while True:
                    conflict_end = -1
                    for interval in sim_usage[r]:
                        int_start, int_end, int_demand = interval
                        if int_start <= t < int_end:
                            if int_end > conflict_end:
                                conflict_end = int_end
                    if conflict_end == -1:
                        break
                    t = conflict_end
                
                if t > earliest_res:
                    earliest_res = t
        
        actual_start = max(earliest_prec, earliest_res)
        sim_starts[current] = actual_start
        
        # Update resource usage
        for r in range(n_resources):
            d = demands[current][r]
            if d > 0:
                sim_usage[r].append((actual_start, actual_start + durations[current], d))
        
        eligible.remove(current)
        for s in successors[current]:
            sim_in_deg[s] -= 1
            if sim_in_deg[s] == 0:
                eligible.add(s)
        
        if time.time() - start_time > DEADLINE:
            break

    # Calculate Regret: how much does activity i delay its successors?
    regret = np.zeros(n)
    for i in range(n):
        finish_i = sim_starts[i] + durations[i]
        for s in successors[i]:
            delay = max(0.0, finish_i - est_prec[s])
            regret[i] += delay * bl[s]
            
    # Normalize components
    max_regret = np.max(regret) if n > 0 else 1.0
    regret_norm = regret / max_regret if max_regret > 0 else regret
    
    bl_norm = bl / np.max(bl) if np.max(bl) > 0 else bl
    
    max_scarcity = np.max(resource_scarcity) if n > 0 else 1.0
    scarcity_norm = resource_scarcity / max_scarcity if max_scarcity > 0 else resource_scarcity
    
    # Weighted combination:
    # - BL (critical path) is the strongest predictor of makespan
    # - Regret captures resource contention impact on successors
    # - Resource Scarcity identifies bottleneck activities
    # Weight scarcity by BL to focus on critical activities with high resource demand
    weighted_scarcity = scarcity_norm * bl_norm
    
    priority_scores = 0.5 * bl_norm + 0.3 * regret_norm + 0.2 * weighted_scarcity
    
    return priority_scores

def _bottom_level(n, durations, predecessors):
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) != n:
        return durations.copy()
    
    bottom_level = np.zeros(n)
    for node in reversed(topo_order):
        if not successors[node]:
            bottom_level[node] = durations[node]
        else:
            bottom_level[node] = durations[node] + max(bottom_level[s] for s in successors[node])
    
    return bottom_level
# EVOLVE-BLOCK-END
