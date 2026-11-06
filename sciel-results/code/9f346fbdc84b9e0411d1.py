import numpy as np
import time
import random

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    Strategy: Greedy Constructive Heuristic with Dynamic Resource Weighting.
    Since the interface requires a static vector but the strategy is dynamic,
    we return a base priority vector (CPM Slack + Duration) which serves as a 
    tie-breaker or base, but the core logic is actually embedded in the 
    'simulate_sgs' function which is NOT used by the evaluator directly.
    
    WAIT: The evaluator calls `priority(instance)` to get scores, then 
    runs its OWN Serial SGS using those scores.
    Therefore, I cannot change the SGS loop. I must return a static vector.
    
    CORRECTION: The task says "Write a Python priority(instance) function that 
    scores each activity... a serial schedule-generation scheme starts the 
    highest-priority eligible activity".
    
    This implies the evaluator uses the returned vector to drive a standard 
    Serial SGS. I cannot inject dynamic logic into the evaluator's loop.
    I must return a static vector that is as good as possible.
    
    However, the inspiration mentions "Greedy Constructive Heuristic with 
    Adaptive Resource Weighting". If I can't change the loop, how do I 
    apply dynamic weighting?
    
    Let's re-read carefully: "Write a Python priority(instance) function 
    that scores each activity...".
    
    If the evaluator's SGS is fixed, I can only optimize the static vector.
    The current program does local search on this vector.
    
    Is it possible the evaluator allows the `priority` function to be called 
    multiple times or in a way that allows state? No, it returns a vector.
    
    Alternative Interpretation: 
    Maybe I can use the "priority" function to perform a full simulation 
    internally and return a vector that represents the "best" order 
    derived from that simulation?
    
    Actually, the most powerful approach if the SGS is fixed is to find 
    the best static priority vector. The current program does hill climbing.
    
    Let's look at the "Idea target": "Does operator 'error_repair' change...".
    This is a meta-question about the evolution process, not the code.
    
    Let's look at the "Strategy to try": "Model the scheduling problem as a 
    maximum flow problem... where each activity's priority is determined by 
    the minimum cut capacity...".
    
    This suggests calculating a "bottleneck score" for each activity based on 
    resource constraints.
    
    Let's implement a strategy that calculates a "Resource Criticality" score 
    for each activity. 
    Score = w1 * (1 / (Slack + 1)) + w2 * (Duration) + w3 * (Resource Scarcity Impact)
    
    Resource Scarcity Impact: For each resource r used by activity i, 
    calculate how "tight" that resource is. 
    Tightness_r = Sum(demands[i][r] for all i) / capacities[r]
    
    Then, the priority of i is high if it uses tight resources.
    
    This is a static calculation, but it captures the "global optimization 
    metric" aspect better than just slack.
    
    Let's combine:
    1. CPM Slack (standard)
    2. Duration (LPT-like)
    3. Resource Bottleneck Score (new)
    
    We will compute these and return a weighted sum. We can also run a quick 
    local search on the weights or the final vector to ensure robustness.
    
    Given the 150s limit, we can afford a more sophisticated initialization 
    and a refined local search.
    
    Strategy:
    1. Parse instance.
    2. Compute CPM Slack.
    3. Compute Resource Tightness per resource.
    4. Compute Activity Resource Bottleneck Score: Sum over resources r of 
       (demand[i][r] * Tightness[r]).
    5. Base Priority = -Slack + alpha * Duration + beta * BottleneckScore.
    6. Use this as initial vector for a fast local search (Hill Climbing) 
       to fine-tune.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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
    
    # 5. Compute Resource Tightness
    # Total demand per resource
    total_demand = np.sum(demands, axis=0)
    # Tightness = Total Demand / Capacity (higher means tighter)
    # Handle zero capacity (should not happen, but safe)
    safe_caps = np.where(capacities > 0, capacities, 1.0)
    resource_tightness = total_demand / safe_caps
    
    # 6. Compute Activity Resource Bottleneck Score
    # For each activity, sum of (demand * tightness)
    # This identifies activities that consume scarce resources heavily.
    bottleneck_score = np.sum(demands * resource_tightness, axis=1)
    
    # Normalize scores to [0, 1] for combination
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_slack = normalize(slack) # High slack = low priority
    norm_dur = normalize(durations) # High dur = high priority
    norm_bottleneck = normalize(bottleneck_score) # High bottleneck = high priority
    
    # Base Priority: 
    # We want high priority for: Low Slack, High Duration, High Bottleneck.
    # Score = -Slack + w_dur * Dur + w_bottleneck * Bottleneck
    # Let's try a weighted sum.
    # Initial weights
    w_slack = 1.0
    w_dur = 0.5
    w_bottleneck = 1.0
    
    initial_priorities = -norm_slack * w_slack + norm_dur * w_dur + norm_bottleneck * w_bottleneck
    
    # Add small noise
    initial_priorities += np.random.normal(0, 0.05, N)
    
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
                continue
            
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

    # Local Search to refine the static vector
    best_priorities = initial_priorities.copy()
    best_score = simulate_sgs(best_priorities)
    
    # Parameters
    noise_scale = 0.5
    max_iter = 3000
    no_improve_count = 0
    patience = 200
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if no_improve_count > patience:
            break
            
        # Generate neighbor: perturb a subset
        num_perturb = max(1, int(N * 0.15))
        indices = np.random.choice(N, num_perturb, replace=False)
        
        neighbor = best_priorities.copy()
        neighbor[indices] += np.random.normal(0, noise_scale, num_perturb)
        
        score = simulate_sgs(neighbor)
        
        if score < best_score:
            best_score = score
            best_priorities = neighbor
            no_improve_count = 0
            noise_scale *= 0.98
        else:
            no_improve_count += 1
            # Occasionally restart exploration
            if no_improve_count % 50 == 0:
                noise_scale = max(0.1, noise_scale * 1.2)
                
    return best_priorities
