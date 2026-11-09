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
        
    # 1. Pre-compute static features for fallback and baseline
    
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
    
    # Duration: Higher is better (longer tasks take more time)
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    feat_dur = durations / max_dur
    
    # Resource Demand: Higher is better
    if n_res > 0:
        max_demand_ratio = np.max(demands / np.maximum(capacities, 1e-9), axis=1)
    else:
        max_demand_ratio = np.zeros(n)
    max_res = np.max(max_demand_ratio) if np.max(max_demand_ratio) > 0 else 1.0
    feat_res = max_demand_ratio / max_res
    
    # Float: Lower is better. So (max_float - float) / max_float
    max_float = np.max(total_float) if np.max(total_float) > 0 else 1.0
    feat_float = (max_float - total_float) / max_float

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

    # Dynamic Look-Ahead Simulation Heuristic
    # Strategy: Run a fast approximate serial schedule using a simple baseline rule
    # to estimate each activity's actual completion time under resource constraints.
    # Return inverted completion times as priority scores.
    
    # Baseline rule: Longest Duration + Critical Path
    baseline_scores = feat_dur + feat_float
    
    # Run baseline simulation to get estimated completion times
    finished = [False] * n
    scheduled = []
    completion_times = np.zeros(n)
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
            
        # Select highest priority using baseline scores
        best_idx = -1
        best_score = -np.inf
        for i in eligible:
            if baseline_scores[i] > best_score:
                best_score = baseline_scores[i]
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
        completion_times[best_idx] = valid_start + durations[best_idx]
        scheduled.append((valid_start, valid_start + durations[best_idx], demands[best_idx]))
        finished[best_idx] = True
        makespan = max(makespan, valid_start + durations[best_idx])
        
        if time.time() > deadline:
            # Time's up, return what we have
            # For unscheduled activities, use their ES as estimated completion
            for i in range(n):
                if not finished[i]:
                    completion_times[i] = es[i] + durations[i]
            # Invert completion times: lower completion time = higher priority
            max_completion = np.max(completion_times) if np.max(completion_times) > 0 else 1.0
            dynamic_scores = (max_completion - completion_times) / max_completion
            return dynamic_scores.tolist()
    
    # Invert completion times: lower completion time = higher priority
    max_completion = np.max(completion_times) if np.max(completion_times) > 0 else 1.0
    dynamic_scores = (max_completion - completion_times) / max_completion
    
    # Optional: Blend with static features for robustness
    # static_scores = 0.5 * feat_float + 0.3 * feat_dur + 0.2 * feat_res
    # blended_scores = 0.7 * dynamic_scores + 0.3 * static_scores
    
    return dynamic_scores.tolist()
