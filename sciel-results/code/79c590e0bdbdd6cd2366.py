"""RCPSP priority via feature-rich local search with optimized SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute rich features (Slack, Duration, Resource Criticality, Contention).
    2. Initialize priority based on a strong composite heuristic.
    3. Use Local Search (Hill Climbing) to refine priority scores
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

    # Precompute predecessor lists as lists of ints for speed
    pred_list = []
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            pred_list.append([int(i) for i in p if 0 <= int(i) < n])
        else:
            pred_list.append([])

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_list[j])

    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)

    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        EF[u] = ES[u] + durations[u]
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]

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

    # --- Feature computation ---
    def normalize(x):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    # Resource tightness: total demand / capacity (higher = tighter)
    # We want to prioritize activities on tight resources.
    resource_tightness = np.zeros(R)
    for r in range(R):
        total_demand = float(np.sum(demands[:, r]))
        if total_demand > 0:
            resource_tightness[r] = total_demand / capacities[r]
        else:
            resource_tightness[r] = 0.0
    
    # Resource Criticality: sum of demand * tightness for each activity
    # Activities using tight resources are more critical.
    resource_criticality = np.sum(demands * resource_tightness, axis=1)
    
    # Contention: How much does this activity compete with its successors?
    # If an activity has high demand on a resource that its successors also need,
    # delaying it might block them. So high contention -> high priority.
    contention = np.zeros(n)
    for j in range(n):
        for v in succs[j]:
            # Sum over resources of min(demand[j,r], demand[v,r]) * tightness[r]
            shared = np.minimum(demands[j], demands[v])
            contention[j] += np.sum(shared * resource_tightness)

    # Normalize features
    feat_slack = normalize(slack)
    feat_rc = normalize(resource_criticality)
    feat_contention = normalize(contention)
    feat_dur = normalize(durations)
    feat_es = normalize(ES)

    # Base priority: composite of features
    # Low slack = high priority (1 - slack)
    # High RC = high priority
    # High contention = high priority
    # Short duration = slightly higher priority (to clear resources)
    # Early ES = slightly lower priority? Or higher? 
    # Usually, we want to schedule critical path items early.
    # Let's stick to: 0.5 * (1-slack) + 0.3 * RC + 0.2 * Contention
    base_priority = (0.5 * (1.0 - feat_slack) + 
                     0.3 * feat_rc + 
                     0.2 * feat_contention)

    # --- Fast SGS Simulator ---
    # Pre-compute demand matrix for fast access
    demands_np = demands
    
    def simulate_sgs(prio_scores):
        """
        Optimized Serial SGS.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        scheduled = np.zeros(n, dtype=bool)
        
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
                # Fallback: should not happen in valid DAG, but safety net
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

            # Candidate times: es_best and finish times of all scheduled activities >= es_best
            # Optimization: Only consider finish times of scheduled activities
            cand_times = [es_best]
            for k in sched_indices:
                ft = finish_t[k]
                if ft >= es_best:
                    cand_times.append(ft)
            
            cand_times = sorted(set(cand_times))
            
            best_start = None
            found = False
            
            # Pre-allocate arrays for scheduled activities for faster NumPy ops
            if sched_indices:
                s_starts = start_t[sched_indices]
                s_ends = finish_t[sched_indices]
                s_demands = demands_np[sched_indices]
            else:
                s_starts = np.array([])
                s_ends = np.array([])
                s_demands = np.zeros((0, R))

            for t in cand_times:
                end_t = t + durations[best]
                
                # Check feasibility
                feasible = True
                
                # Identify check points: t and any start_k in [t, end_t)
                # We check load at these points.
                check_points = [t]
                for idx in range(len(s_starts)):
                    if t <= s_starts[idx] < end_t:
                        check_points.append(s_starts[idx])
                
                # Unique check points (sorted)
                check_points = sorted(set(check_points))
                
                for tp in check_points:
                    # Calculate load at tp
                    # Active if start_k <= tp < end_k
                    if len(s_starts) > 0:
                        mask = (s_starts <= tp) & (s_ends > tp)
                        current_load = s_demands[mask].sum(axis=0)
                    else:
                        current_load = np.zeros(R)
                    
                    # Add best activity
                    current_load += demands_np[best]
                    
                    if np.any(current_load > capacities + 1e-9):
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    found = True
                    break

            if not found:
                # This should theoretically not happen if we consider all relevant times,
                # but if it does, we have a problem. Fallback to es_best (might violate resources)
                # or return inf. For SGS, we must find a valid time.
                # If no valid time found in cand_times, it means our cand_times were insufficient?
                # No, cand_times includes all start/end events. 
                # If it fails, it means the activity can't start at es_best or any subsequent event.
                # This implies infeasibility or a bug. 
                # However, SGS guarantees a solution if the instance is feasible.
                # Let's just use es_best as a fallback to avoid crash, though it might be invalid.
                best_start = es_best

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            scheduled[best] = True
            sched_indices.append(best)
            remaining.remove(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Search Strategy ---
    # Start with the composite heuristic
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Local Search: Multi-start with Hill Climbing
    # We will run several local search runs from different initializations
    
    num_runs = 5
    for run in range(num_runs):
        if time.time() - start_time > DEADLINE - 1.0:
            break
        
        # Initialize
        if run == 0:
            current_scores = best_scores.copy()
        else:
            # Random initialization or perturbed best
            if random.random() < 0.5:
                current_scores = np.random.rand(n)
            else:
                current_scores = best_scores + np.random.randn(n) * 0.1
        
        current_ms = simulate_sgs(current_scores)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 200 # Reduced to allow more runs
        
        while time.time() - start_time < DEADLINE - 1.0 and no_improve_count < max_no_improve:
            # Generate neighbor: Swap two random activities
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
