"""RCPSP priority via nonlinear multiplicative scoring with fast SGS and Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque
from heapq import heappush, heappop


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute rich features: slack, duration, resource criticality, downstream count.
    2. Use a nonlinear multiplicative priority: duration^a * (1/slack)^b * crit^c * down^d
    3. Optimize exponents (a, b, c, d) via Nelder-Mead on makespan.
    4. Fast event-based SGS simulation.
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

    # --- Extract instance components ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

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

    # Build pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_set[j])

    q = deque([i for i in range(n) if in_degree[i] == 0])
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

    slack = LS - ES

    # --- Compute base features ---
    eps = 1e-9
    max_demand_ratio = np.zeros(n)
    for i in range(n):
        for r in range(R):
            if capacities[r] > 0:
                ratio = demands[i, r] / capacities[r]
                if ratio > max_demand_ratio[i]:
                    max_demand_ratio[i] = ratio

    # Compute downstream count via BFS from each node
    downstream_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for v in succs[node]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        downstream_count[i] = len(visited)

    # Normalize features to avoid overflow/underflow in power operations
    dur_norm = durations / (np.max(durations) + eps)
    slack_inv = 1.0 / (slack + eps)
    slack_inv_norm = slack_inv / (np.max(slack_inv) + eps)
    crit_norm = max_demand_ratio / (np.max(max_demand_ratio) + eps)
    down_norm = downstream_count / (np.max(downstream_count) + eps)

    # --- Fast SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS using event-driven approach for speed.
        """
        if n == 0:
            return 0.0
        
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        # Track unscheduled activities
        unscheduled = set(range(n))
        
        # Pre-compute eligible check: for each activity, count unscheduled predecessors
        unscheduled_preds_count = np.zeros(n, dtype=int)
        for j in range(n):
            unscheduled_preds_count[j] = len(pred_set[j])
        
        # Priority heap for eligible activities (negative priority for max-heap)
        eligible_heap = []
        
        # Initially, activities with no predecessors are eligible
        for j in range(n):
            if unscheduled_preds_count[j] == 0:
                heappush(eligible_heap, (-prio_scores[j], j))
        
        while unscheduled:
            if not eligible_heap:
                # Should not happen if instance is valid, but fallback
                for j in unscheduled:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            # Get highest priority eligible activity
            neg_prio, j = heappop(eligible_heap)
            
            if j not in unscheduled:
                continue  # stale entry
            
            # Compute earliest start from precedence
            es_j = 0.0
            for p in pred_set[j]:
                cand = start_t[p] + durations[p]
                if cand > es_j:
                    es_j = cand
            
            # Find earliest feasible start time >= es_j
            # Candidate times: es_j and all finish times of currently scheduled activities >= es_j
            cand_times = {es_j}
            for k in range(n):
                if finish_t[k] != np.inf and finish_t[k] >= es_j:
                    cand_times.add(finish_t[k])
            
            cand_times = sorted(cand_times)
            
            best_start = None
            for t in cand_times:
                if t < es_j - 1e-9:
                    continue
                
                end_t = t + durations[j]
                
                # Check resource feasibility: for each resource, compute usage in [t, end_t)
                feasible = True
                for r in range(R):
                    # Compute total demand from activities active in [t, end_t)
                    usage = 0.0
                    for k in range(n):
                        if start_t[k] != np.inf:
                            # Activity k is active in [start_t[k], finish_t[k])
                            # Check overlap with [t, end_t)
                            if start_t[k] < end_t and finish_t[k] > t:
                                usage += demands[k, r]
                    
                    if usage + demands[j, r] > capacities[r] + 1e-9:
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    break
            
            if best_start is None:
                best_start = es_j
            
            start_t[j] = best_start
            finish_t[j] = best_start + durations[j]
            unscheduled.discard(j)
            
            # Update eligibility of successors
            for v in succs[j]:
                unscheduled_preds_count[v] -= 1
                if unscheduled_preds_count[v] == 0 and v in unscheduled:
                    heappush(eligible_heap, (-prio_scores[v], v))
        
        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Nonlinear multiplicative scoring ---
    def compute_scores(exponents):
        """
        Compute priority scores using nonlinear multiplicative formula:
        score = dur_norm^a * slack_inv_norm^b * crit_norm^c * down_norm^d
        """
        a, b, c, d = exponents
        scores = (dur_norm ** a) * (slack_inv_norm ** b) * (crit_norm ** c) * (down_norm ** d)
        return scores

    # --- Nelder-Mead optimization for exponents ---
    # Initial simplex: 4 exponents (a, b, c, d)
    # Start with reasonable values
    initial_point = np.array([1.0, 2.0, 1.0, 0.5])
    
    # Create initial simplex
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    # Simplex vertices
    simplex = [initial_point.copy()]
    for i in range(4):
        vertex = initial_point.copy()
        step = 0.5 if initial_point[i] == 0 else initial_point[i] * 0.5
        vertex[i] += step
        simplex.append(vertex)
    
    # Evaluate initial simplex
    simplex_scores = []
    for vertex in simplex:
        scores = compute_scores(vertex)
        ms = simulate_sgs(scores)
        simplex_scores.append(ms)
    
    # Track best
    best_idx = int(np.argmin(simplex_scores))
    best_vertex = simplex[best_idx].copy()
    best_ms = simplex_scores[best_idx]
    
    # Nelder-Mead iterations
    max_iter = 1000
    tol = 1e-6
    
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        
        # Sort simplex by function value
        order = np.argsort(simplex_scores)
        simplex = [simplex[i] for i in order]
        simplex_scores = [simplex_scores[i] for i in order]
        
        # Check convergence
        if abs(simplex_scores[0] - simplex_scores[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid + alpha * (centroid - simplex[-1])
        reflected_scores = compute_scores(reflected)
        reflected_ms = simulate_sgs(reflected_scores)
        
        if simplex_scores[0] <= reflected_ms < simplex_scores[-2]:
            # Accept reflection
            simplex[-1] = reflected
            simplex_scores[-1] = reflected_ms
        elif reflected_ms < simplex_scores[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded_scores = compute_scores(expanded)
            expanded_ms = simulate_sgs(expanded_scores)
            
            if expanded_ms < reflected_ms:
                simplex[-1] = expanded
                simplex_scores[-1] = expanded_ms
            else:
                simplex[-1] = reflected
                simplex_scores[-1] = reflected_ms
        else:
            # Contraction
            if reflected_ms < simplex_scores[-1]:
                # Outside contraction
                contracted = centroid + rho * (reflected - centroid)
            else:
                # Inside contraction
                contracted = centroid - rho * (simplex[-1] - centroid)
            
            contracted_scores = compute_scores(contracted)
            contracted_ms = simulate_sgs(contracted_scores)
            
            if contracted_ms < simplex_scores[-1]:
                simplex[-1] = contracted
                simplex_scores[-1] = contracted_ms
            else:
                # Shrink
                for i in range(1, 5):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex_scores[i] = simulate_sgs(compute_scores(simplex[i]))
        
        # Update best
        current_best_idx = int(np.argmin(simplex_scores))
        if simplex_scores[current_best_idx] < best_ms:
            best_ms = simplex_scores[current_best_idx]
            best_vertex = simplex[current_best_idx].copy()
    
    # Final priority scores
    best_scores = compute_scores(best_vertex)
    return best_scores
# EVOLVE-BLOCK-END
