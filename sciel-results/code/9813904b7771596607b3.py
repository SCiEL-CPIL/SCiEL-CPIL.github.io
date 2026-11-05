import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Hill-Climbing local search on priority vectors, initialized with CPM slack.
    """
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    deadline = start_time + 140.0
    
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
             
    # Compute CPM Slack for initialization
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
                
    # 2. Forward Pass (Earliest Start/Finish)
    ES = np.zeros(N)
    EF = np.zeros(N)
    for u in topo_order:
        max_pred_ef = 0
        for p in preds[u]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[u] = max_pred_ef
        EF[u] = ES[u] + durations[u]
        
    # 3. Backward Pass (Latest Start/Finish)
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
        
    # 4. Slack = LS - ES
    slack = LS - ES
    
    # Initialize priorities with negative slack (low slack = high priority)
    # Add small noise to break ties and allow exploration
    initial_priorities = -slack + np.random.normal(0, 0.1, N)
    
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
        
        # Pre-compute eligible check logic for speed
        # We can maintain a count of unscheduled predecessors
        unsched_preds = np.array([len(p) for p in preds])
        
        # Initial eligible
        eligible_set = set(i for i in range(N) if unsched_preds[i] == 0)
        
        while not np.all(scheduled):
            # If no eligible, wait
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
                        # Update eligibility of successors
                        for v in succs[idx]:
                            if not scheduled[v]:
                                unsched_preds[v] -= 1
                                if unsched_preds[v] == 0:
                                    eligible_set.add(v)
                active = still_active
                continue
            
            # Sort eligible by priority (descending)
            # Convert to list for sorting
            elig_list = list(eligible_set)
            elig_list.sort(key=lambda x: priorities[x], reverse=True)
            
            any_scheduled = False
            # Try to schedule as many as possible in priority order
            to_remove = []
            for i in elig_list:
                # Check resources
                if np.all(remaining_res >= demands[i]):
                    # Schedule
                    scheduled[i] = True
                    remaining_res -= demands[i]
                    finish_t = time_current + durations[i]
                    active.append((finish_t, i))
                    makespan = max(makespan, finish_t)
                    any_scheduled = True
                    to_remove.append(i)
                    
            # Update eligible set
            if to_remove:
                for i in to_remove:
                    eligible_set.remove(i)
            
            if not any_scheduled:
                # Couldn't schedule anyone, wait
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

    # Local Search: Hill Climbing
    best_priorities = initial_priorities.copy()
    best_score = simulate_sgs(best_priorities)
    
    # Parameters
    noise_scale = 1.0
    max_iter = 2000 # Limit iterations to ensure time safety
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        # Perturb a subset of activities (e.g., 20% of N)
        num_perturb = max(1, int(N * 0.2))
        indices = np.random.choice(N, num_perturb, replace=False)
        
        neighbor = best_priorities.copy()
        neighbor[indices] += np.random.normal(0, noise_scale, num_perturb)
        
        score = simulate_sgs(neighbor)
        
        if score < best_score:
            best_score = score
            best_priorities = neighbor
            # Reduce noise slightly to refine
            noise_scale *= 0.95
        else:
            # If no improvement, occasionally reset noise scale to explore
            if random.random() < 0.05:
                noise_scale = max(0.1, noise_scale * 1.5)
                
    return best_priorities
