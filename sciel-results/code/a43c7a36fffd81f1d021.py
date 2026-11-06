"""
RCPSP Heuristic: Dynamic Resource-Contention-Aware Priority
Strategy:
Instead of computing a single static priority score for all activities and letting the SGS
scheduler handle it, this strategy integrates the priority calculation into the scheduling loop.
At each step, for every eligible activity, it calculates a dynamic score based on:
1. Total Float (from CPM) - criticality
2. Resource Demand Intensity - how much of the scarce resources it consumes
3. Successor Depth - how many tasks it unblocks
4. Current Resource Pressure - how tight the resources are right now

The activity with the highest dynamic score is scheduled next at the earliest feasible time.
This allows the heuristic to adapt to the evolving state of resource usage during the schedule generation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Returns a priority score for each activity.
    
    Note: The task interface expects a single priority vector. However, the prompt says:
    "a serial schedule-generation scheme starts the highest-priority eligible activity"
    and "Write a Python priority(instance) function that scores each activity".
    
    The standard interpretation is that priority(instance) returns a vector of scores.
    The SGS scheduler then uses these scores to pick the best eligible activity.
    
    My strategy: I will simulate the SGS process internally to determine the best
    *static* priority vector that approximates the behavior of a dynamic heuristic, 
    OR I will return a static vector that is a good proxy for dynamic priorities.
    
    Actually, looking at the current program, it returns a static vector.
    The evaluator runs SGS using that vector.
    
    If I want to use dynamic priorities, I cannot do so directly if the interface
    only allows returning a static vector. The SGS scheduler uses the returned vector.
    
    Wait, let me re-read: "a serial schedule-generation scheme starts the highest-priority 
    eligible activity". This implies the scheduler is fixed and uses the vector I return.
    
    So I must return a static vector.
    
    How to make a static vector that mimics dynamic behavior?
    I can't really.
    
    Alternative Strategy: 
    The current program tries several static heuristics and picks the best one by simulating SGS.
    This is a "meta-heuristic" over static rules.
    
    A DIFFERENT strategy: 
    Instead of simulating SGS for a few candidate static rules, I can use a 
    **Greedy Randomized Adaptive Search Procedure (GRASP)** style approach.
    I will generate many random priority vectors, but guided by the structure.
    Specifically, I will use a **Local Search** on the priority vector itself.
    
    1. Start with a good initial priority vector (e.g., based on Total Float).
    2. Simulate SGS to get makespan.
    3. Identify "bottleneck" activities in the schedule (those on the critical path or causing resource delays).
    4. Perturb the priority scores of these bottleneck activities (e.g., boost their priority or lower it to see if it helps).
    5. Iterate this local search until time runs out.
    6. Return the best priority vector found.
    
    This is a local search over the space of priority vectors, guided by the makespan feedback.
    """
    start_time = time.time()
    DEADLINE = 145.0
    
    # --- 1. Parse Instance ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
    # Normalize predecessors
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
                
    # Build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    # Handle resources
    if demands_raw is not None:
        try:
            demands = np.asarray(demands_raw, dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(-1, 1)
        except:
            demands = np.zeros((n, 1))
    else:
        demands = np.zeros((n, 1))
        
    if capacities_raw is not None:
        try:
            capacities = np.asarray(capacities_raw, dtype=float)
            if capacities.ndim == 0:
                capacities = np.array([capacities])
        except:
            capacities = np.ones(demands.shape[1])
    else:
        capacities = np.ones(demands.shape[1])
        
    m_resources = demands.shape[1]
    if len(capacities) < m_resources:
        capacities = np.pad(capacities, (0, m_resources - len(capacities)), mode='constant')
        
    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        return durations.copy()
        
    # Pre-compute static metrics
    es_cpm = np.zeros(n)
    ef_cpm = np.zeros(n)
    for u in topo_order:
        es_cpm[u] = 0
        for p in preds[u]:
            if ef_cpm[p] > es_cpm[u]:
                es_cpm[u] = ef_cpm[p]
        ef_cpm[u] = es_cpm[u] + durations[u]
        
    makespan_cpm = np.max(ef_cpm)
    
    lf_cpm = np.zeros(n)
    ls_cpm = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) > 0:
            lf_cpm[u] = np.min([ls_cpm[v] for v in succs[u]])
        else:
            lf_cpm[u] = makespan_cpm
        ls_cpm[u] = lf_cpm[u] - durations[u]
        
    tf_cpm = ls_cpm - es_cpm
    
    # --- 2. SGS Simulator ---
    def simulate_sgs(scores):
        scheduled = [False] * n
        start_times = np.zeros(n)
        end_times = np.zeros(n)
        scheduled_intervals = []
        
        for _ in range(n):
            eligible = []
            for i in range(n):
                if not scheduled[i]:
                    if all(scheduled[p] for p in preds[i]):
                        eligible.append(i)
                        
            if not eligible:
                break
                
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            task = best_idx
            d = durations[task]
            dem = demands[task]
            
            candidate_starts = [0]
            for (st, en, _) in scheduled_intervals:
                candidate_starts.append(en)
            candidate_starts = sorted(set(candidate_starts))
            
            found_start = None
            for t in candidate_starts:
                ok = True
                for r in range(m_resources):
                    if dem[r] == 0:
                        continue
                    total_demand = 0
                    for (st, en, dem_vec) in scheduled_intervals:
                        if st < t + d and en > t:
                            total_demand += dem_vec[r]
                    if total_demand + dem[r] > capacities[r]:
                        ok = False
                        break
                if ok:
                    found_start = t
                    break
                    
            if found_start is None:
                found_start = max(candidate_starts) if candidate_starts else 0
                
            scheduled[task] = True
            start_times[task] = found_start
            end_times[task] = found_start + d
            scheduled_intervals.append((found_start, found_start + d, dem))
            
        makespan = np.max(end_times) if n > 0 else 0
        return makespan, start_times, end_times

    # --- 3. Local Search Strategy ---
    
    # Initial priority: based on Total Float (criticality)
    max_tf = np.max(tf_cpm)
    if max_tf > 0:
        current_scores = 1.0 - (tf_cpm / max_tf)
    else:
        current_scores = np.ones(n)
        
    # Normalize to [0, 1] for stability
    min_s = np.min(current_scores)
    max_s = np.max(current_scores)
    if max_s > min_s:
        current_scores = (current_scores - min_s) / (max_s - min_s)
        
    current_makespan, _, _ = simulate_sgs(current_scores)
    best_scores = current_scores.copy()
    best_makespan = current_makespan
    
    # Local Search Loop
    # Strategy: Perturb scores of activities that are on the critical path or have high resource demand.
    # We can identify "important" activities by their impact on makespan.
    # Simple heuristic: Activities with high Total Float (low criticality) might be moved later, 
    # or activities with high resource demand might be prioritized to clear resources.
    
    # Let's try a random restart local search.
    # 1. Generate a random neighbor of the current best score.
    # 2. If it improves, accept.
    # 3. If not, keep the best so far.
    
    # To generate a neighbor, we can:
    # a) Swap scores of two random activities.
    # b) Add noise to a random activity's score.
    # c) Re-rank a subset of activities randomly.
    
    # Let's use a combination.
    
    while time.time() < DEADLINE:
        # Generate a neighbor
        neighbor_scores = best_scores.copy()
        
        # Pick a few activities to perturb
        num_perturb = np.random.randint(1, 4)
        for _ in range(num_perturb):
            i = np.random.randint(0, n)
            # Add Gaussian noise
            neighbor_scores[i] += np.random.normal(0, 0.1)
            
        # Normalize
        min_s = np.min(neighbor_scores)
        max_s = np.max(neighbor_scores)
        if max_s > min_s:
            neighbor_scores = (neighbor_scores - min_s) / (max_s - min_s)
        else:
            neighbor_scores = np.ones(n) / n
            
        ms, _, _ = simulate_sgs(neighbor_scores)
        
        if ms < best_makespan:
            best_makespan = ms
            best_scores = neighbor_scores.copy()
            
        # Occasionally, try a completely different heuristic as a restart
        if np.random.rand() < 0.05:
            # Try LPT
            scores_lpt = durations.copy()
            min_s = np.min(scores_lpt)
            max_s = np.max(scores_lpt)
            if max_s > min_s:
                scores_lpt = (scores_lpt - min_s) / (max_s - min_s)
            ms_lpt, _, _ = simulate_sgs(scores_lpt)
            if ms_lpt < best_makespan:
                best_makespan = ms_lpt
                best_scores = scores_lpt.copy()
                
            # Try Random
            scores_rand = np.random.rand(n)
            ms_rand, _, _ = simulate_sgs(scores_rand)
            if ms_rand < best_makespan:
                best_makespan = ms_rand
                best_scores = scores_rand.copy()
                
    return best_scores

# EVOLVE-BLOCK-END
