"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit

def _check_deadline():
    return _time.monotonic() >= _DEADLINE

def _fast_greedy_score(ready_operations, time):
    """
    Fast heuristic to rank operations for beam selection or final tie-breaking.
    Uses a combination of remaining work urgency and duration speed.
    """
    n = len(ready_operations)
    if n == 0:
        return np.array([])
    
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        op = ready_operations[i]
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_d = min(durations) if durations else 1.0
        # Heuristic: Prefer high urgency (rw) and low duration (1/min_d)
        # Invert to minimize
        if min_d < 1e-9:
            min_d = 1e-9
        scores[i] = rw / min_d
    return scores

def _simulate_one_step(ready_operations, chosen_idx, time):
    """
    Simulates the completion of one operation and returns the new ready_operations list.
    This is a simplified simulation assuming no other events happen simultaneously 
    and machine availability is not a bottleneck for the *next* immediate ready set 
    (or we approximate it). 
    Note: In a real JSS, machine availability matters. 
    However, 'ready_operations' implies they are currently schedulable.
    When we pick one, the others remain ready? 
    Usually, in event-driven JSS, if a machine is free, multiple jobs might be ready.
    If we pick one, it occupies a machine. The others wait?
    The prompt says `dispatch` picks which ready operation to *start*.
    This implies the others are NOT started yet.
    So the state after starting `chosen_idx`:
    1. `chosen_idx` is removed from ready.
    2. Its job's `remaining_work` decreases by `duration`.
    3. If the job has more ops, the *next* op of that job becomes ready? 
       No, the next op becomes ready only after the *current* one finishes.
       But we are *starting* it now. It is not finished.
       So the next op is NOT ready yet.
    4. So the new ready set is just the old ready set minus the chosen one?
       Wait, if we start an op, it is "in progress". It is no longer "ready".
       The other ready ops are still ready (waiting for machines).
       
    BUT, the simulation logic in the evaluator is fixed.
    If I pick index i, the evaluator will:
    - Mark op i as started.
    - Update time? No, it's event driven.
    - The next call to dispatch will happen when *some* event occurs (e.g., a machine becomes free).
    
    If I want to lookahead, I need to know what the *next* state looks like.
    The next state depends on when the *current* chosen op finishes AND when other ops finish.
    This makes pure lookahead difficult without knowing the full schedule.
    
    HOWEVER, we can approximate.
    If we assume that after starting op `i`, the *next* dispatch call will happen after the *shortest* currently running or just-started op finishes.
    
    Let's simplify the Lookahead:
    Instead of simulating time, we simulate the *impact on the Critical Path*.
    
    Revised Lookahead Strategy:
    1. For each candidate i, calculate `new_rw_i` (remaining work of job i after this op).
    2. Calculate `new_global_max` (max remaining work among all jobs, assuming job i's work is reduced).
    3. For the "Second Step", assume we will pick the best candidate from the *remaining* ready ops (plus potentially the next op of job i if it finished instantly, which it doesn't).
    4. Actually, since job i is now *in progress*, it is not ready.
    5. The next decision will be made on the *other* ready ops.
    6. So, for candidate i, the "future" state is determined by how we handle the *other* ready ops.
    7. Score(i) = Immediate_Cost(i) + Alpha * Future_Cost(Other_Ops)
    
    Future_Cost(Other_Ops) can be estimated by the best possible move among the others.
    This is essentially:
    Score(i) = Cost(i) + Min_{j != i} Cost(j, given i is gone)
    
    This is a 2-step Min-Max heuristic.
    """
    pass # Implementation below in dispatch

def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # 1. Pre-compute features for all ops
    # We need: remaining_work, min_duration, remaining_ops
    # We also need to know the "Next Op" info? No, we only have current ready ops.
    
    remaining_works = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    remaining_ops_counts = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        op = ready_operations[i]
        remaining_works[i] = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_durations[i] = min(durations) if durations else 1.0
        remaining_ops_counts[i] = op.get("remaining_ops", 0)

    # 2. Beam Search Setup
    # Select top K candidates based on a fast greedy score
    # Greedy Score: Minimize (remaining_work / min_duration) -> Actually, we want to prioritize high urgency?
    # Let's use the same logic as before for the "greedy" base:
    # Minimize new_global_max * min_duration
    
    current_max_rw = np.max(remaining_works)
    
    def calc_greedy_score(idx):
        rw_i = remaining_works[idx]
        dur_i = min_durations[idx]
        if dur_i < 1e-9: dur_i = 1e-9
        new_rw_i = rw_i - dur_i
        # Calculate max of others
        # Optimization: Pre-compute sorted remaining works
        pass

    # Let's just use the fast greedy score to pick the top 3
    # Score to minimize: remaining_work / min_duration (Urgency * Speed)
    # Or the previous composite: new_global_max * duration
    
    # Let's stick to the previous composite for the base score to ensure we don't regress
    # on the "island best" logic, but add the lookahead.
    
    sorted_indices = np.argsort(-remaining_works)
    max_idx = sorted_indices[0]
    second_max_idx = sorted_indices[1] if n > 1 else -1
    max_val = remaining_works[max_idx]
    second_max_val = remaining_works[second_max_idx] if second_max_idx != -1 else 0.0

    base_scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        rw_i = remaining_works[i]
        duration_i = min_durations[i]
        if duration_i < 1e-9:
            duration_i = 1e-9
            
        new_rw_i = rw_i - duration_i
        
        if rw_i == max_val:
            if i == max_idx:
                if second_max_val == max_val:
                    max_others = max_val
                else:
                    max_others = second_max_val
            else:
                max_others = max_val
        else:
            max_others = max_val
            
        new_global_max = max(new_rw_i, max_others)
        
        ops_left = remaining_ops_counts[i]
        completion_bonus = 0.9 if ops_left == 1 else 1.0
        
        base_scores[i] = new_global_max * duration_i * completion_bonus

    # Select Top 3 candidates with lowest base score
    K = min(3, n)
    top_k_indices = np.argsort(base_scores)[:K]
    
    # 3. Lookahead Evaluation for Top K
    # For each candidate i in top_k:
    #    Assume we pick i.
    #    The "remaining" problem is for the other n-1 ops.
    #    The "cost" of the future is approximated by the best possible move among the others.
    #    Future_Cost(i) = Min_{j != i} Base_Score(j, context=i)
    #    Note: Base_Score(j) might change if i was the max, but for a quick lookahead,
    #    we can reuse the base scores of the others as a proxy for their future impact.
    #    Total_Score(i) = Base_Score(i) + Lambda * Future_Cost(i)
    
    Lambda = 0.5 # Weight of future consideration
    
    best_total_score = np.inf
    best_idx = top_k_indices[0]
    
    for i in top_k_indices:
        current_score = base_scores[i]
        
        # Estimate future cost: The best move among the others
        # We want to minimize the future makespan contribution.
        # The "best" future move is the one with the lowest base score among the remaining.
        # However, if we pick i, the "remaining" set is all j != i.
        # The cost of the system after i is roughly driven by the max of the remaining works.
        # A simple proxy: The minimum base score of the remaining ops.
        
        min_future_score = np.inf
        for j in range(n):
            if j == i:
                continue
            # We could recalculate base_scores[j] in the context of i being removed,
            # but that's expensive. Using pre-computed base_scores[j] is a good approximation.
            if base_scores[j] < min_future_score:
                min_future_score = base_scores[j]
        
        total_score = current_score + Lambda * min_future_score
        
        if total_score < best_total_score:
            best_total_score = total_score
            best_idx = i

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
