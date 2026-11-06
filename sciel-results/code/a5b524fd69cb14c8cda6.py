"""RCPSP priority via local search with optimized SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute features (Slack, Duration, Resource Criticality).
    2. Initialize priority based on Minimum Slack (strong heuristic).
    3. Use Hill Climbing / Simulated Annealing to refine priority scores
       by swapping or perturbing scores, evaluated via a fast SGS simulation.
    """
    start_time = time.time()
    DEADLINE = 118.0

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components ---
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
            idx = int(i)
            if 0 <= idx < n:
                in_degree[j] += 1

    # Use a simple list as queue for topo sort
    q = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]

    slack = LS - ES

    # --- Structural Change: Resource Bottleneck Centrality ---
    # Calculate the "resource pressure" of the subgraph rooted at each activity.
    # This serves as the initial priority score.
    # Score_i = Sum over all activities j in {i} U Descendants(i) of (Sum over resources r demanded by j of 1/Capacity_r)
    
    # First, compute the "resource cost" of each individual activity
    # cost[j] = sum(1/capacity[r] for r in resources demanded by j)
    # Note: If capacity is 0, treat as high cost? Assume > 0.
    inv_caps = 1.0 / np.where(capacities > 0, capacities, 1.0) # Shape (R,)
    
    # demands is (n, R)
    # individual_cost[j] = sum(demands[j, r] * inv_caps[r] for r in R) 
    # Wait, the directive says "sum of inverse resource capacities". 
    # Usually, demand matters. If demand is 0, it doesn't use the resource.
    # Let's assume "inverse resource capacities weighted by demand" or just "inverse capacity" if used.
    # "sum of inverse resource capacities" usually implies the resource constraint tightness.
    # Let's use: cost[j] = sum( (demands[j, r] > 0) * (1/capacity[r]) ) 
    # Or better: cost[j] = sum( demands[j, r] / capacity[r] ) -- this is the fraction of capacity used.
    # The directive says "sum of inverse resource capacities". It doesn't explicitly mention demand magnitude, 
    # but "resource bottleneck" implies how much it competes. 
    # Let's stick to the directive's wording strictly if possible, but "weighted by number of downstream activities" 
    # applies to the sum.
    # Let's define node_weight[j] = sum( 1/capacity[r] for r where demands[j,r] > 0 )
    
    node_weights = np.zeros(n)
    for j in range(n):
        # Find resources used by j
        used_res = np.where(demands[j] > 0)[0]
        if len(used_res) > 0:
            node_weights[j] = np.sum(inv_caps[used_res])
        else:
            node_weights[j] = 0.0
            
    # Now, propagate these weights to predecessors.
    # Total_Centrality[i] = node_weights[i] + sum(Total_Centrality[v] for v in Successors(i))
    # This is a backward pass in topological order (reverse topo).
    
    centrality = node_weights.copy()
    for u in reversed(topo_order):
        for v in succs[u]:
            centrality[u] += centrality[v]
            
    # Normalize centrality to [0, 1] for priority scores
    min_c = np.min(centrality)
    max_c = np.max(centrality)
    if max_c > min_c:
        base_priority = (centrality - min_c) / (max_c - min_c)
    else:
        base_priority = np.zeros(n)
        
    # Higher centrality -> Higher Priority

    # --- Optimized SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Optimized Serial SGS.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        scheduled = np.zeros(n, dtype=bool)
        
        pred_list = [list(p) for p in preds]
        
        remaining = list(range(n))
        sched_indices = [] # List of indices of scheduled activities
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf')

            # Find eligible
            eligible = []
            for j in remaining:
                is_eligible = True
                for p in pred_list[j]:
                    if not scheduled[p]:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(j)

            if not eligible:
                # Fallback: shouldn't happen in valid DAG
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                    scheduled[j] = True
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_list[best]:
                cand = finish_t[p]
                if cand > es_best:
                    es_best = cand

            # Candidate times: es_best and finish times of scheduled activities >= es_best
            # We check these in increasing order.
            cand_times = [es_best]
            for k in sched_indices:
                if finish_t[k] >= es_best:
                    cand_times.append(finish_t[k])
            
            cand_times = sorted(set(cand_times))
            
            best_start = es_best
            found = False
            
            # Pre-compute arrays for scheduled activities to speed up checks
            if sched_indices:
                s_starts = start_t[sched_indices]
                s_ends = finish_t[sched_indices]
                s_demands = demands[sched_indices]
                n_scheduled = len(sched_indices)
            else:
                s_starts = np.array([])
                s_ends = np.array([])
                s_demands = np.zeros((0, R))
                n_scheduled = 0

            for t in cand_times:
                end_t = t + durations[best]
                
                # Check feasibility
                # We need to ensure that for all time points in [t, end_t), 
                # load <= capacity.
                # The load function changes only at start and end times of scheduled activities.
                # Since we are adding `best`, we check if `load_existing + demand_best <= capacity`.
                
                # Breakpoints to check: t, and any start_k in (t, end_t)
                # Note: We don't need to check end_k because load drops there.
                
                check_points = [t]
                for idx in range(n_scheduled):
                    if t < s_starts[idx] < end_t:
                        check_points.append(s_starts[idx])
                
                # t is always in check_points.
                # We can just iterate through check_points.
                
                feasible = True
                for tp in check_points:
                    # Calculate load at tp
                    # Active if s_starts <= tp < s_ends
                    if n_scheduled > 0:
                        mask = (s_starts <= tp) & (s_ends > tp)
                        current_load = s_demands[mask].sum(axis=0)
                    else:
                        current_load = np.zeros(R)
                    
                    # Add best activity's demand
                    current_load += demands[best]
                    
                    if np.any(current_load > capacities + 1e-9):
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    found = True
                    break

            if not found:
                best_start = es_best

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            scheduled[best] = True
            sched_indices.append(best)
            remaining.remove(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Search Strategy ---
    # Start with Resource Bottleneck Centrality
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Local Search with Random Restarts
    # Given the strong initialization, we focus on refining.
    
    max_time_per_run = (DEADLINE - (time.time() - start_time)) / 5.0 # 5 runs
    
    for run in range(5):
        if time.time() - start_time > DEADLINE - 1.0:
            break
        
        # Initialize
        if run == 0:
            current_scores = best_scores.copy()
        else:
            # Perturbed best to escape local optima
            current_scores = best_scores + np.random.randn(n) * 0.2
        
        current_ms = simulate_sgs(current_scores)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 1000 # Increased limit as simulation is faster
        
        while time.time() - start_time < DEADLINE - 1.0 and no_improve_count < max_no_improve:
            # Generate neighbor: Swap two random activities
            # Or swap adjacent in topological order? Random is safer.
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], current_scores[i1]
            
            neighbor_ms = simulate_sgs(neighbor_scores)
            
            if neighbor_ms < current_ms:
                current_scores = neighbor_scores
                current_ms = neighbor_ms
                no_improve_count = 0
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_scores = current_scores.copy()
            else:
                no_improve_count += 1
                
    return best_scores
# EVOLVE-BLOCK-END
