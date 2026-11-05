"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = _time.monotonic() + 115.0  # safety margin before 120s limit

def _check_deadline():
    return _time.monotonic() >= _DEADLINE

def dispatch(ready_operations, time):
    if _check_deadline():
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Pre-compute base features
    remaining_works = np.empty(n, dtype=np.float64)
    min_durations = np.empty(n, dtype=np.float64)
    max_durations = np.empty(n, dtype=np.float64)
    remaining_ops_counts = np.empty(n, dtype=np.float64)
    
    job_ids = [op["job"] for op in ready_operations]
    for i in range(n):
        op = ready_operations[i]
        remaining_works[i] = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if durations:
            min_durations[i] = min(durations)
            max_durations[i] = max(durations)
        else:
            min_durations[i] = 0.0
            max_durations[i] = 0.0
        remaining_ops_counts[i] = op.get("remaining_ops", 0)

    # Strategy: "Urgency x Blocking" Composite Score
    # 1. Urgency: How much remaining work does this job have? (Higher is more urgent to start)
    # 2. Speed: How fast can this operation be completed? (Faster is better, frees machine)
    # 3. Blocking: Does this job have many remaining ops? (If yes, starting it early might be good to get it moving, 
    #    but if it has few, it's almost done, so finish it quickly).
    
    # Let's define a composite score.
    # We want to maximize the "impact" of starting this operation.
    # Impact = (Reduction in Critical Path) * (Probability of Machine Availability) ? 
    # We don't have machine load info directly.
    
    # Let's look at the inspirations:
    # "urgency * speed" product.
    # Urgency = remaining_work (normalized?)
    # Speed = 1 / duration (or just duration inverted)
    
    # Let's try a non-linear composite:
    # Score = (remaining_work ** alpha) * (1 / min_duration ** beta) * (1 / remaining_ops ** gamma)
    
    # However, remaining_work is absolute. We should probably normalize or use differences.
    # But simple products often work well in heuristics if scales are similar.
    
    # Let's refine the "Critical Path Reduction" idea from inspiration 2, but make it a product.
    # The current program uses a linear sum with a penalty.
    # Let's try: Score = -New_Global_Max * (min_duration / max_possible_duration)
    # This penalizes picking a long operation even if it reduces the max, because it keeps the machine busy.
    # Wait, if we pick a long operation, the machine is busy longer. If we pick a short one, the machine frees up sooner.
    # So we want to minimize New_Global_Max AND minimize min_duration.
    # Multiplication: Minimize (New_Global_Max * min_duration).
    # This creates a trade-off. A very large reduction in New_Global_Max might justify a slightly longer duration.
    
    # Let's compute New_Global_Max for each candidate as before.
    
    current_max_rw = np.max(remaining_works)
    sorted_indices = np.argsort(-remaining_works)
    max_idx = sorted_indices[0]
    second_max_idx = sorted_indices[1] if n > 1 else -1
    max_val = remaining_works[max_idx]
    second_max_val = remaining_works[second_max_idx] if second_max_idx != -1 else 0.0

    best_score = np.inf  # We want to minimize
    best_idx = 0
    
    for i in range(n):
        rw_i = remaining_works[i]
        duration_i = min_durations[i]
        
        # Avoid division by zero or very small numbers
        if duration_i < 1e-9:
            duration_i = 1e-9
            
        new_rw_i = rw_i - duration_i
        
        # Determine max_others
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
        
        # Composite Score: Product of "Resulting Critical Path" and "Operation Duration"
        # Minimizing this balances reducing the makespan lower bound and keeping machines free.
        # We also add a small penalty for jobs with many remaining ops? 
        # If a job has many remaining ops, completing one doesn't finish it.
        # If a job has 1 remaining op, completing it finishes the job.
        # Finishing a job removes it from the critical path potentially.
        # Let's add a bonus for completing a job.
        
        ops_left = remaining_ops_counts[i]
        # If ops_left is 1, this op completes the job.
        # Give a small discount to the score if this completes the job.
        completion_bonus = 1.0
        if ops_left == 1:
            completion_bonus = 0.9  # 10% discount for completing a job
            
        score = new_global_max * duration_i * completion_bonus
        
        if score < best_score:
            best_score = score
            best_idx = i

    if _check_deadline():
        return 0

    return best_idx
# EVOLVE-BLOCK-END
