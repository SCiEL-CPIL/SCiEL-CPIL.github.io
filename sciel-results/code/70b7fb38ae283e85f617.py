import numpy as np
import time as time_module
import random
from collections import defaultdict

def _soft_max(values, temperature=1.0):
    """Smooth max via log-sum-exp for differentiable-like surrogate."""
    if not values:
        return 0.0
    vals = np.array(values, dtype=np.float64)
    max_val = np.max(vals)
    if max_val < -1e10:
        return max_val
    shifted = vals - max_val
    exp_sum = np.sum(np.exp(shifted / temperature))
    return max_val + temperature * np.log(exp_sum)

def _soft_min(values, temperature=1.0):
    """Smooth min via log-sum-exp for differentiable-like surrogate."""
    if not values:
        return 0.0
    vals = np.array(values, dtype=np.float64)
    min_val = np.min(vals)
    if min_val > 1e10:
        return min_val
    shifted = vals - min_val
    exp_sum = np.sum(np.exp(-shifted / temperature))
    return min_val - temperature * np.log(exp_sum)

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 145.0  # 5s safety margin before 150s kill
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # --- HUNGARIAN-INSPIRED SLACK-TIME GREEDY ---
    # Treat as bipartite job-machine assignment. For each ready op, compute
    # the minimum slack across its available machines (earliest start - current time).
    # Prioritize ops with larger minimum slack to avoid blocking critical machines.
    
    # Precompute features
    features = []
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        min_dur = min(durations) if durations else 0
        max_dur = max(durations) if durations else 0
        avg_dur = sum(durations) / len(durations) if durations else 0
        
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        # Slack estimate: how much time until this op could start on each machine
        # In open-shop, all machines are available, so slack = 0 for all.
        # Instead, use a proxy: ops with fewer machine options are more constrained.
        # Minimum slack proxy = -n_machines (fewer options = more constrained = higher priority)
        slack_proxy = -n_machines
        
        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "min_dur_inv": 1.0 / (min_dur + 1.0),
            "n_rem_ops": n_rem,
            "n_machines": n_machines,
            "slack_proxy": slack_proxy,
            "critical_path": rw / max(n_machines, 1),
            "job": op.get("job", -1),
            "op": op.get("op", -1),
        })
    
    # Check if we can simulate forward
    can_simulate = False
    if ready_operations:
        sample_op = ready_operations[0]
        rem_ops = sample_op.get("remaining_ops", None)
        if isinstance(rem_ops, list) and len(rem_ops) > 0:
            if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
                can_simulate = True
    
    if can_simulate and n <= 12:
        # Improved beam search with smoother makespan estimator
        def simulate_one_step(ready_ops, t, chosen_idx):
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
        
        def estimate_makespan_smooth(ready_ops, current_time, temp=2.0):
            """Smooth makespan estimate using soft-max over job completion times."""
            if not ready_ops:
                return current_time
            
            # Group by job to estimate per-job completion
            job_work = defaultdict(float)
            for op in ready_ops:
                job = op.get("job", 0)
                job_work[job] += op["remaining_work"]
            
            if not job_work:
                return current_time
            
            # Per-job completion estimate: current_time + remaining work for that job
            # (rough: assumes sequential within job, parallel across jobs)
            completion_times = [current_time + w for w in job_work.values()]
            
            # Use soft-max to get a smooth upper bound on makespan
            return _soft_max(completion_times, temperature=temp)
        
        # Beam search with width 4, depth 2
        beam_width = 4
        candidates = []
        for i in range(n):
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, i)
            score = estimate_makespan_smooth(new_ready, new_t)
            candidates.append((score, i))
        
        if not candidates:
            return int(np.argmax([f["remaining_work"] for f in features]))
        
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:beam_width]
        
        best_score = float('inf')
        best_idx = top_k[0][1]
        
        for level1_score, level1_idx in top_k:
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, level1_idx)
            
            for j in range(len(new_ready)):
                if time_module.time() > deadline:
                    break
                new_ready2, new_t2 = simulate_one_step(new_ready, new_t, j)
                score2 = estimate_makespan_smooth(new_ready2, new_t2)
                # Smooth combination: use soft-max of level1 and level2 scores
                total_score = _soft_max([level1_score, score2], temperature=1.0)
                if total_score < best_score:
                    best_score = total_score
                    best_idx = level1_idx
        
        return best_idx
    
    # --- FALLBACK: Slack-time greedy with weighted features ---
    # Primary signal: slack proxy (constrained ops first)
    # Secondary: remaining work (long jobs first)
    # Tertiary: few remaining ops (nearly done jobs first)
    
    scores = []
    for f in features:
        # Slack-time greedy: prioritize ops with fewer machine options (more constrained)
        slack_score = f["slack_proxy"] * 5.0  # -n_machines, so fewer options = more negative = higher priority
        
        # Weighted combination
        s = (slack_score + 
             0.8 * f["remaining_work"] + 
             0.3 * f["n_rem_ops"] * 10.0 + 
             0.2 * f["critical_path"])
        scores.append(s)
    
    return int(np.argmax(scores))
