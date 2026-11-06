import numpy as np
import time as time_module
import random
from collections import defaultdict

# Global deadline tracking for the evaluation
_START_TIME = None
_DEADLINE = None

def _get_deadline():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time_module.time()
        _DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s kill
    return _DEADLINE

def _time_ok():
    return time_module.time() < _get_deadline()

def _soft_max(values, beta=10.0):
    """Log-sum-exp approximation to max, scaled by beta for sharpness."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    m = np.max(arr)
    if m == 0:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta

def _estimate_makespan(ready_ops, current_time, job_remaining_work=None):
    """
    Estimate makespan using per-job critical paths and smooth max.
    In open-shop, jobs can be processed in parallel on different machines,
    but each job's operations must be sequential.
    """
    if not ready_ops:
        return current_time
    
    if job_remaining_work is None:
        job_remaining_work = defaultdict(float)
        for op in ready_ops:
            jid = op["job"]
            job_remaining_work[jid] = max(job_remaining_work.get(jid, 0), op["remaining_work"])
    
    if job_remaining_work:
        job_works = list(job_remaining_work.values())
        smax = _soft_max(job_works, beta=15.0)
        return current_time + smax
    return current_time

def _simulate_one_step(ready_ops, t, chosen_idx):
    """
    Simulate executing chosen_idx, return new ready ops and new time.
    In open-shop, only the chosen operation is processed; others remain ready.
    """
    op = ready_ops[chosen_idx]
    durations = list(op["machine_options"].values())
    min_dur = min(durations) if durations else 0
    comp_t = t + min_dur
    
    new_ready = []
    for j, other in enumerate(ready_ops):
        if j == chosen_idx:
            rem = op.get("remaining_ops", [])
            if isinstance(rem, list) and len(rem) > 0:
                next_op = rem[0].copy()
                next_op["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                if isinstance(rem, list) and len(rem) > 1:
                    next_op["remaining_ops"] = rem[1:]
                else:
                    next_op["remaining_ops"] = []
                new_ready.append(next_op)
        else:
            new_ready.append(other)
    return new_ready, comp_t

def _beam_search_dispatch(ready_operations, time, beam_width=4, depth=2):
    """
    Beam search over dispatch decisions.
    Returns the index of the best operation to dispatch.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # First level: evaluate each candidate
    candidates = []
    for i in range(n):
        if not _time_ok():
            break
        new_ready, new_t = _simulate_one_step(ready_operations, time, i)
        score = _estimate_makespan(new_ready, new_t)
        candidates.append((score, i))
    
    if not candidates:
        return 0
    
    # Sort by score, keep beam_width
    candidates.sort(key=lambda x: x[0])
    top_k = candidates[:beam_width]
    
    if depth <= 1 or not _time_ok():
        return top_k[0][1]
    
    # Second level: for each top candidate, look one more step
    best_score = float('inf')
    best_idx = top_k[0][1]
    
    for level1_score, level1_idx in top_k:
        if not _time_ok():
            break
        new_ready, new_t = _simulate_one_step(ready_operations, time, level1_idx)
        
        if not new_ready:
            # All done, this is a great choice
            total_score = level1_score
            if total_score < best_score:
                best_score = total_score
                best_idx = level1_idx
            continue
        
        # From this state, try each possible next choice
        for j in range(len(new_ready)):
            if not _time_ok():
                break
            new_ready2, new_t2 = _simulate_one_step(new_ready, new_t, j)
            score2 = _estimate_makespan(new_ready2, new_t2)
            # The makespan is the maximum completion time across all paths
            # Use soft max to combine level1 and level2 scores
            total_score = _soft_max([level1_score, score2], beta=20.0)
            if total_score < best_score:
                best_score = total_score
                best_idx = level1_idx
    
    return best_idx

def _hungarian_greedy_dispatch(ready_operations, time):
    """
    Hungarian-algorithm-inspired greedy heuristic.
    Treats the problem as a bipartite job-machine assignment.
    Selects the ready operation that maximizes the minimum slack time
    across its available machines, while also considering remaining work.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Precompute features
    features = []
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        
        # Slack time: earliest possible start minus current time
        # In open-shop, all machines are available, so slack is 0
        # But we can use min_dur as a proxy for how "urgent" this op is
        slack = min_dur  # Lower slack means more urgent
        
        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "n_rem_ops": n_rem,
            "n_machines": n_machines,
            "slack": slack,
            "critical_path": rw / max(n_machines, 1)
        })
    
    # Score each operation using a weighted composite
    # Weight towards operations with high remaining work (to finish jobs)
    # and low slack (to avoid blocking)
    scores = []
    for f in features:
        # Normalize remaining work
        s = 1.0 * f["remaining_work"] + \
            0.1 * (1.0 / (f["min_dur"] + 1.0)) * f["remaining_work"] + \
            0.3 * f["n_rem_ops"] * 10.0 + \
            0.4 * f["critical_path"]
        scores.append(s)
    
    return int(np.argmax(scores))

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Try beam search first if feasible
    if n <= 10 and _time_ok():
        # Check if we can simulate
        can_simulate = False
        if ready_operations:
            sample_op = ready_operations[0]
            rem_ops = sample_op.get("remaining_ops", None)
            if isinstance(rem_ops, list) and len(rem_ops) > 0:
                if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
                    can_simulate = True
        
        if can_simulate:
            return _beam_search_dispatch(ready_operations, time, beam_width=4, depth=2)
    
    # Fallback to Hungarian-inspired greedy
    return _hungarian_greedy_dispatch(ready_operations, time)
