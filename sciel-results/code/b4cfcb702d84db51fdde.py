import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]
    
    n = len(durations)
    n_res = len(capacities)
    
    if n == 0:
        return []
        
    # 1. Pre-compute static features
    
    # Successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Forward Pass (ES, EF)
    es = np.zeros(n)
    ef = np.zeros(n)
    for i in range(n):
        for p in predecessors[i]:
            es[i] = max(es[i], ef[p])
        ef[i] = es[i] + durations[i]
    
    project_end_static = np.max(ef)
    
    # Backward Pass (LS, LF)
    ls = np.zeros(n)
    lf = np.zeros(n)
    for i in range(n-1, -1, -1):
        if not successors[i]:
            lf[i] = project_end_static
        else:
            lf[i] = min(ls[s] for s in successors[i])
        ls[i] = lf[i] - durations[i]
        
    total_float = ls - es
    tail = project_end_static - ef
    
    # Depth (Longest path from start) - often correlates with criticality
    depth = np.zeros(n)
    for i in range(n):
        if predecessors[i]:
            depth[i] = max(depth[p] for p in predecessors[i])
        depth[i] += durations[i] # This is actually EF if started at 0, but let's use path length in nodes or time
        # Actually, let's use "Level" or "Rank"
    # Rank: 0 for sources, 1 + max(rank(preds)) for others
    rank = np.zeros(n, dtype=int)
    for i in range(n):
        if predecessors[i]:
            rank[i] = 1 + max(rank[p] for p in predecessors[i])
            
    # Resource Features
    # Max demand ratio per activity (0 to 1)
    if n_res > 0:
        max_demand_ratio = np.max(demands / np.maximum(capacities, 1e-9), axis=1)
        # Sum of demand ratios
        sum_demand_ratio = np.sum(demands / np.maximum(capacities, 1e-9), axis=1)
    else:
        max_demand_ratio = np.zeros(n)
        sum_demand_ratio = np.zeros(n)
        
    # Normalization
    # We want features in [0, 1] where 1 is "more important" (should be scheduled earlier)
    
    # Float: Lower is better. So (max_float - float) / max_float
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    feat_float = (max_float - total_float) / max_float
    
    # Tail: Higher is better.
    max_tail = np.max(tail) if np.max(tail) > 0 else 1.0
    feat_tail = tail / max_tail
    
    # Duration: Higher is better (longer tasks take more time, scheduling them early might help avoid bottlenecks? Or is it neutral? Usually, long tasks are critical).
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    feat_dur = durations / max_dur
    
    # Resource Demand: Higher is better (resource-intensive tasks should be scheduled when resources are likely available? Or avoid peak? 
    # Actually, in RCPSP, scheduling high-demand tasks early can cause congestion. 
    # However, "Critical" tasks often have high demand. 
    # Let's try both: High demand (feat_res) and Low demand (1 - feat_res) might be useful. 
    # Let's stick to High demand for now, as it's often correlated with criticality.
    max_res = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    feat_res = max_demand_ratio / max_res
    
    # Rank: Higher is better (deeper in the network)
    max_rank = np.max(rank) if np.max(rank) > 0 else 1.0
    feat_rank = rank.astype(float) / max_rank
    
    # Successor Count: Higher is better (bottlenecks)
    succ_count = np.array([len(successors[i]) for i in range(n)])
    max_succ = np.max(succ_count) if np.max(succ_count) > 0 else 1.0
    feat_succ = succ_count.astype(float) / max_succ

    # Feature Matrix
    # Columns: float, tail, dur, res, rank, succ
    features = np.column_stack((feat_float, feat_tail, feat_dur, feat_res, feat_rank, feat_succ))
    
    # Simulation Function
    def simulate(scores):
        """
        Simulates serial schedule generation.
        scores: array of priority scores (higher is better).
        Returns makespan.
        """
        finished = [False] * n
        # scheduled: list of (start, end, res_vector)
        scheduled = []
        makespan = 0
        
        for _ in range(n):
            # Find eligible
            eligible = []
            for i in range(n):
                if finished[i]:
                    continue
                if all(finished[p] for p in predecessors[i]):
                    eligible.append(i)
            
            if not eligible:
                break
                
            # Select highest priority
            # Tie-breaker: lower index
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
            # Find earliest start time
            es_pred = es[best_idx]
            
            # Candidate times: es_pred and end times of scheduled activities > es_pred
            candidate_times = {es_pred}
            for (s, e, d_vec) in scheduled:
                if e > es_pred:
                    candidate_times.add(e)
            
            valid_start = None
            for t in sorted(candidate_times):
                # Check resource usage in [t, t + durations[best_idx]]
                usage = np.zeros(n_res)
                for (s, e, d_vec) in scheduled:
                    if s < t + durations[best_idx] and e > t:
                        usage += d_vec
                if np.all(usage + demands[best_idx] <= capacities):
                    valid_start = t
                    break
            
            if valid_start is None:
                # Should not happen if capacities are sufficient
                valid_start = es_pred
            
            # Schedule activity
            scheduled.append((valid_start, valid_start + durations[best_idx], demands[best_idx]))
            finished[best_idx] = True
            makespan = max(makespan, valid_start + durations[best_idx])
            
            if time.time() > deadline:
                return makespan
                
        return makespan

    # Search for best weights
    best_makespan = float('inf')
    best_scores = None
    
    # Initial set of weights
    # Weights are for: [float, tail, dur, res, rank, succ]
    
    # 1. Critical Path (Max Float Score)
    w1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 2. Critical Path + Tail
    w2 = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    
    # 3. Critical Path + Duration
    w3 = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
    
    # 4. Critical Path + Resource
    w4 = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    
    # 5. Rank based
    w5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    
    # 6. Successor based
    w6 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    initial_weights = [w1, w2, w3, w4, w5, w6]
    
    # Add random weights
    np.random.seed(42)
    for _ in range(20):
        w = np.random.rand(6)
        w /= np.sum(w)
        initial_weights.append(w)
        
    # Evaluate initial weights
    candidates = []
    for w in initial_weights:
        scores = features @ w
        m = simulate(scores)
        if m < best_makespan:
            best_makespan = m
            best_scores = scores
        candidates.append((m, w))
        
    # Local Search
    # Sort by makespan
    candidates.sort(key=lambda x: x[0])
    
    best_w = candidates[0][1]
    
    # Perform hill climbing
    iterations = 0
    max_iterations = 500
    step_size = 0.1
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
            
        improved = False
        for _ in range(10): # 10 random perturbations per iteration
            # Perturb best_w
            w_pert = best_w + np.random.normal(0, step_size, 6)
            w_pert = np.clip(w_pert, 0, None)
            if np.sum(w_pert) == 0:
                continue
            w_pert /= np.sum(w_pert)
            
            scores = features @ w_pert
            m = simulate(scores)
            
            if m < best_makespan:
                best_makespan = m
                best_scores = scores
                best_w = w_pert
                improved = True
                # Reduce step size slightly if improved?
                # step_size *= 0.9 
                break # Restart perturbations from new best
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 0.01:
                break
                
        iterations += 1

    if best_scores is None:
        # Fallback
        best_scores = features @ w1

    return best_scores.tolist()
