import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Constructive heuristic based on Resource Pressure and CPM Slack.
    1. Compute CPM Slack.
    2. Compute Resource Pressure (Demand / Capacity).
    3. Combine into a static priority score.
    4. Run a lightweight local search (hill climbing) to fine-tune the static scores
       by swapping pairs of activities in the priority list.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Extract and process instance data
    N = len(instance["durations"])
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)  # Shape (N, R)
    capacities = np.array(instance["capacities"], dtype=float) # Shape (R,)
    
    # Handle predecessors
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
             
    # Compute CPM Slack
    # 1. Topological Sort
    in_degree = [len(p) for p in preds]
    q = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # 2. Forward Pass
    ES = np.zeros(N)
    EF = np.zeros(N)
    for u in topo_order:
        max_pred_ef = 0
        for p in preds[u]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[u] = max_pred_ef
        EF[u] = ES[u] + durations[u]
        
    # 3. Backward Pass
    project_end = max(EF)
    LF = np.zeros(N)
    LS = np.zeros(N)
    for u in reversed(topo_order):
        min_succ_lf = project_end
        for v in succs[u]:
            if LF[v] < min_succ_lf:
                min_succ_lf = LF[v]
        LF[u] = min_succ_lf
        LS[u] = LF[u] - durations[u]
        
    slack = LS - ES

    # Compute Resource Pressure
    # For each activity, calculate the max ratio of demand to capacity across resources.
    # This identifies activities that are most "constrained" by resources.
    # Avoid division by zero
    safe_caps = np.where(capacities == 0, 1e-9, capacities)
    ratios = demands / safe_caps
    resource_pressure = np.max(ratios, axis=1)
    
    # Normalize slack and pressure to [0, 1] for combination
    # Low slack is good (high priority). High pressure is "bad" in terms of availability, 
    # but in scheduling, we often want to schedule high pressure items early to avoid bottlenecks?
    # Actually, the "Longest Processing Time" (LPT) or "Most Immediate Successors" (MIS) are common.
    # Let's try a combination: High Priority = Low Slack + High Resource Pressure (to clear bottleneck early).
    
    # Normalize
    slack_min, slack_max = np.min(slack), np.max(slack)
    if slack_max > slack_min:
        norm_slack = (slack - slack_min) / (slack_max - slack_min)
    else:
        norm_slack = np.zeros(N)
        
    press_min, press_max = np.min(resource_pressure), np.max(resource_pressure)
    if press_max > press_min:
        norm_press = (resource_pressure - press_min) / (press_max - press_min)
    else:
        norm_press = np.zeros(N)

    # We want High Score = High Priority.
    # Low Slack -> High Priority. So use (1 - norm_slack).
    # High Pressure -> High Priority (to satisfy bottleneck early). So use norm_press.
    
    # Weights: Slack is usually the dominant factor in RCPSP heuristics.
    # Let's try 0.7 * Slack + 0.3 * Pressure
    initial_priorities = 0.7 * (1 - norm_slack) + 0.3 * norm_press

    def simulate_sgs(priorities):
        """
        Fast Serial SGS simulation.
        Returns makespan.
        """
        scheduled = np.zeros(N, dtype=bool)
        active = []  # List of (finish_time, activity_index)
        remaining_res = capacities.copy()
        time_current = 0.0
        makespan = 0.0
        
        unsched_preds = np.array([len(p) for p in preds])
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)
        
        while not np.all(scheduled):
            if not eligible_set:
                if not active:
                    break # Deadlock/Error
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue
            
            # Sort eligible by priority (descending)
            elig_list = list(eligible_set)
            elig_list.sort(key=lambda x: priorities[x], reverse=True)
            
            any_scheduled = False
            to_remove = []
            for i in elig_list:
                if np.all(remaining_res >= demands[i]):
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    to_remove.append(i)
                    
            if to_remove:
                for i in to_remove:
                    eligible_set.remove(i)
            
            if not any_scheduled:
                if not active:
                    break
                min_finish = min(f for f, _ in active)
                time_current = min_finish
                still_active = []
                for f, idx in active:
                    if f > time_current:
                        still_active.append((f, idx))
                    else:
                        remaining_res += demands[idx]
                        makespan = max(makespan, f)
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
        
        return makespan

    best_priorities = initial_priorities.copy()
    best_score = simulate_sgs(best_priorities)
    
    # Local Search: Swap-based Hill Climbing
    # This is structurally different from random noise perturbation.
    # It explores the space of permutations more efficiently by swapping pairs.
    
    max_iter = 5000
    no_improve_count = 0
    max_no_improve = 100
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        if no_improve_count > max_no_improve:
            break
            
        # Pick two random activities to swap
        i, j = np.random.choice(N, 2, replace=False)
        
        neighbor = best_priorities.copy()
        neighbor[i], neighbor[j] = best_priorities[j], best_priorities[i]
        
        score = simulate_sgs(neighbor)
        
        if score < best_score:
            best_score = score
            best_priorities = neighbor
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_priorities
