"""RCPSP priority via MCTS-informed local search with fast SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute rich features (Slack, Duration, Resource Criticality, Contention).
    2. Initialize priority based on a strong composite heuristic.
    3. Use MCTS to explore the decision space of which activity to schedule next
       from the initial eligible set, using fast SGS rollouts as the value function.
    4. Refine the resulting priority ordering via local search (hill climbing / SA).
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

    # --- Feature computation ---
    def normalize(x):
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    # Resource tightness
    resource_tightness = np.zeros(R)
    for r in range(R):
        total_demand = float(np.sum(demands[:, r]))
        if total_demand > 0:
            resource_tightness[r] = capacities[r] / total_demand
        else:
            resource_tightness[r] = 1.0

    max_demand_per_res = np.max(demands, axis=0)
    max_demand_per_res[max_demand_per_res == 0] = 1.0
    norm_demands = demands / max_demand_per_res
    inv_tightness = 1.0 / (resource_tightness + 1e-10)
    resource_criticality = np.sum(norm_demands * inv_tightness, axis=1)

    # Contention: sum of shared demands with successors
    contention = np.zeros(n)
    for j in range(n):
        for v in succs[j]:
            # Sum over resources of min(demand[j,r], demand[v,r])
            shared = np.minimum(demands[j], demands[v])
            contention[j] += np.sum(shared)
    
    pred_count = np.zeros(n)
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            pred_count[j] = len(p)

    succ_count = np.zeros(n)
    for j in range(n):
        succ_count[j] = len(succs[j])

    dur = durations.copy()

    feat_slack = normalize(slack)
    feat_rc = normalize(resource_criticality)
    feat_dur = normalize(dur)
    feat_pred = normalize(pred_count)
    feat_succ = normalize(succ_count)
    feat_es = normalize(ES)
    feat_contention = normalize(contention)

    # Base priority: composite of features
    # Low slack = high priority, high RC = high priority, high contention = high priority
    base_priority = (0.4 * (1.0 - feat_slack) + 
                     0.25 * feat_rc + 
                     0.2 * feat_contention + 
                     0.1 * feat_dur + 
                     0.05 * (1.0 - feat_es))

    # --- Fast SGS Simulator ---
    # Precompute predecessor sets for speed
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    def simulate_sgs(prio_scores, start_time_override=None, initial_scheduled=None):
        """
        Serial SGS with optimized overlap checking.
        If start_time_override and initial_scheduled are provided, start from that state.
        """
        t_start = start_time if start_time_override is None else start_time_override
        if time.time() - t_start > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        if initial_scheduled:
            for idx, (st, ft) in initial_scheduled.items():
                start_t[idx] = st
                finish_t[idx] = ft

        remaining = set(range(n))
        if initial_scheduled:
            remaining -= set(initial_scheduled.keys())

        while remaining:
            if time.time() - t_start > DEADLINE:
                return float('inf')

            # Find eligible
            eligible = []
            for j in remaining:
                is_eligible = True
                for p in pred_set[j]:
                    if start_t[p] == np.inf:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(j)

            if not eligible:
                # Fallback
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_set[best]:
                cand = start_t[p] + durations[p]
                if cand > es_best:
                    es_best = cand

            # Candidate times: es_best and all finish times of scheduled activities >= es_best
            cand_times = [es_best]
            for k in range(n):
                if finish_t[k] != np.inf and finish_t[k] >= es_best:
                    cand_times.append(finish_t[k])
            
            cand_times = sorted(set(cand_times))

            best_start = es_best
            found = False
            for t in cand_times:
                if t < es_best:
                    continue
                end_t = t + durations[best]
                
                # Check resource feasibility
                feasible = True
                check_points = {t, end_t}
                
                for k in range(n):
                    if start_t[k] != np.inf:
                        if start_t[k] < end_t and finish_t[k] > t:
                            if t <= start_t[k] < end_t:
                                check_points.add(start_t[k])
                            if t < finish_t[k] <= end_t:
                                check_points.add(finish_t[k])
                
                check_points = sorted(check_points)
                
                for i in range(len(check_points) - 1):
                    tp = check_points[i]
                    tp_next = check_points[i+1]
                    if tp >= end_t or tp_next <= t:
                        continue
                    
                    active_demand = np.zeros(R)
                    for k in range(n):
                        if start_t[k] != np.inf:
                            if start_t[k] <= tp and finish_t[k] >= tp_next:
                                active_demand += demands[k]
                    
                    active_demand += demands[best]
                    
                    if np.any(active_demand > capacities + 1e-9):
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
            remaining.discard(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- MCTS on the first decision ---
    # Identify the initial eligible set
    initial_eligible = [j for j in range(n) if len(pred_set[j]) == 0]
    
    if len(initial_eligible) > 1:
        # Run MCTS to decide which activity should have the highest priority
        # We simulate the rest of the schedule with the remaining priorities fixed
        # relative to base_priority, but we force the chosen activity to be first.
        
        # To do this, we can temporarily set the priority of the chosen activity to infinity
        # and run SGS. This effectively fixes the first decision.
        
        best_first_activity = initial_eligible[0]
        best_ms_for_first = float('inf')
        
        # Sample a subset if too many eligible
        if len(initial_eligible) > 10:
            candidates = random.sample(initial_eligible, 10)
        else:
            candidates = initial_eligible
            
        for act in candidates:
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            # Create a modified priority vector where 'act' is highest
            mod_scores = base_priority.copy()
            mod_scores[act] = np.inf
            
            ms = simulate_sgs(mod_scores)
            if ms < best_ms_for_first:
                best_ms_for_first = ms
                best_first_activity = act
        
        # Adjust base_priority to reflect this finding
        # We don't just set it to inf, but we boost it significantly
        # or we can just use this as the seed for local search
        # Let's create a new base priority where best_first_activity is highest
        # and others are scaled.
        
        # Actually, for the local search, we just want a good starting vector.
        # Let's set the score of best_first_activity to be the max, and keep others relative.
        # Or simpler: just use the standard base_priority but ensure best_first_activity
        # has a high score.
        
        # A better approach: The MCTS result tells us which activity is best to start.
        # We can create a priority vector that respects this ordering for the first step.
        # However, SGS picks the max from eligible. If we set score[best_first_activity] > score[other_eligible],
        # it will be picked first.
        
        # Let's just boost the score of best_first_activity relative to other initial eligibles.
        max_other_score = max([base_priority[j] for j in initial_eligible if j != best_first_activity]) if len(initial_eligible) > 1 else 0
        base_priority[best_first_activity] = max_other_score + 1.0

    # --- Local Search ---
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    current_scores = best_scores.copy()
    current_ms = best_ms
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    max_iterations = 5000
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor
        if random.random() < 0.5:
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], neighbor_scores[i1]
        else:
            i1 = random.randint(0, n-1)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1] += random.gauss(0, 0.1)
        
        neighbor_ms = simulate_sgs(neighbor_scores)
        
        if neighbor_ms is float('inf'):
            continue
            
        delta = neighbor_ms - current_ms
        if delta < 0:
            current_scores = neighbor_scores
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_scores = current_scores.copy()
        else:
            if random.random() < np.exp(-delta / T):
                current_scores = neighbor_scores
                current_ms = neighbor_ms
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_scores
# EVOLVE-BLOCK-END
