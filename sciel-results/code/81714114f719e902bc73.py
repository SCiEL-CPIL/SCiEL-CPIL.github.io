"""Job-Shop Scheduling Dispatch Rule.
Combines Most Work Remaining (MWL), Longest Processing Time (LPT) for bottleneck
elimination, and Urgency (fewest remaining ops) to minimize makespan.
Returns (index, best_machine) to allow immediate machine assignment.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.monotonic()
TIME_BUDGET = 115.0  # Safety margin before 120s/150s limit


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Greedy Bottleneck Elimination (LPT) combined with Criticality (MWL).
    1. Prioritize operations with high remaining work (Criticality).
    2. Prioritize operations with longer minimum processing times (Bottleneck/LPT) 
       to prevent long inflexible jobs from becoming critical path bottlenecks later.
    3. Prioritize operations with fewer remaining steps (Urgency) to complete jobs.
    4. For the selected operation, assign the machine with the shortest duration (SPT)
       to free up resources quickly.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo, key=mo.get)
            return (0, best_m)
        return 0

    # Deadline check: fallback to fast heuristic if time is running out
    if _elapsed() > TIME_BUDGET - 5.0:
        # Fallback: MWL + LPT simple
        scores = []
        for op in ready_operations:
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            min_d = min(mo.values()) if mo else 0
            # Simple combined score
            scores.append(rw * 1.0 + min_d * 0.5)
        best_idx = int(np.argmax(scores))
        mo = ready_operations[best_idx].get("machine_options", {})
        if mo:
            best_m = min(mo, key=mo.get)
            return (best_idx, best_m)
        return best_idx

    # Extract features
    remaining_works = np.array([op.get("remaining_work", 0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([max(op.get("remaining_ops", 1), 1) for op in ready_operations], dtype=np.float64)
    
    min_durations = np.zeros(n, dtype=np.float64)
    best_machines = [None] * n
    num_options = np.zeros(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            durs = np.array(list(mo.values()), dtype=np.float64)
            min_durations[i] = durs.min()
            best_machines[i] = min(mo, key=mo.get)
            num_options[i] = len(mo)
        else:
            min_durations[i] = 0.0
            best_machines[i] = None
            num_options[i] = 0.0

    # Normalization
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_md = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    max_ro = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    max_no = np.max(num_options) if np.max(num_options) > 0 else 1.0

    # Component Scores (all normalized to [0, 1], higher is better)
    
    # 1. Criticality: Most Work Remaining
    score_work = remaining_works / max_rw
    
    # 2. Bottleneck/LPT: Longer minimum duration is better (prioritize long ops)
    # This helps clear critical path items early.
    score_bottleneck = min_durations / max_md
    
    # 3. Urgency: Fewer remaining ops is better
    # Normalize so that 1 op = 1.0, max ops = 0.0
    score_urgency = 1.0 - (remaining_ops / max_ro)
    
    # 4. Flexibility: More options is slightly better (less contention risk?)
    # Actually, in JSS, fewer options often means higher priority (constrained).
    # But for makespan, flexibility might help avoid bottlenecks. 
    # Let's keep it small or remove. The inspirations didn't emphasize it heavily.
    # Let's use "Constrainedness": Fewer options = Higher Priority (more critical).
    # If max_no is 1, score is 1. If max_no is 5, score is 0.2.
    # score_flex = num_options / max_no  # More flex = higher score
    # Let's try: score_constrained = 1.0 - (num_options / max_no)
    # But if all have 1 option, score is 0. 
    # Let's stick to the main 3 components for stability.

    # Weights:
    # MWL is generally the strongest predictor.
    # LPT (Bottleneck) is the new strategic addition.
    # Urgency helps with job completion.
    
    w_work = 0.45
    w_bottleneck = 0.35
    w_urgency = 0.20
    
    scores = (w_work * score_work) + (w_bottleneck * score_bottleneck) + (w_urgency * score_urgency)
    
    best_idx = int(np.argmax(scores))
    best_machine = best_machines[best_idx]
    
    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx

# EVOLVE-BLOCK-END
