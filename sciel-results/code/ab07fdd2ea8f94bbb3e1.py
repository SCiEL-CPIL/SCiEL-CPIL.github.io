import numpy as np
import time as time_module
import math

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Deadline guard
    deadline = time_module.time() + 140.0
    
    # Precompute features
    features = []
    for op in ready_operations:
        rw = op["remaining_work"]
        durations = list(op["machine_options"].values())
        if not durations:
            durations = [0]
        min_dur = min(durations)
        max_dur = max(durations)
        avg_dur = sum(durations) / len(durations)
        
        rem_ops = op.get("remaining_ops", [])
        n_rem = len(rem_ops) if isinstance(rem_ops, list) else 0
        
        n_machines = len(op["machine_options"]) if op["machine_options"] else 1
        
        features.append({
            "remaining_work": rw,
            "min_dur": min_dur,
            "max_dur": max_dur,
            "avg_dur": avg_dur,
            "min_dur_inv": 1.0 / (min_dur + 1.0),
            "n_rem_ops": n_rem,
            "n_machines": n_machines,
            "critical_path": rw / max(n_machines, 1),
            # LPT-inspired: prioritize long, inflexible tasks
            "lpt_score": min_dur,
            # Flexibility: more machine options = more flexible
            "flexibility": n_machines,
        })
    
    # --- LPT-inspired dispatch with smooth makespan proxy ---
    # The key insight: in open-shop scheduling, selecting the operation with the
    # largest minimum processing time (LPT) early helps balance machine loads
    # and prevents late-stage bottlenecks where a single long operation dominates.
    #
    # We combine LPT priority with a smooth (log-sum-exp) makespan estimate to
    # get a differentiable surrogate that balances:
    #   1. Longest processing time (to avoid tail bottlenecks)
    #   2. Remaining work per job (to complete critical jobs)
    #   3. Machine load balance (to avoid overloading one machine)
    
    # Check if we can do a deeper lookahead
    can_simulate = False
    if n <= 15:
        sample_op = ready_operations[0]
        rem_ops = sample_op.get("remaining_ops", None)
        if isinstance(rem_ops, list) and len(rem_ops) > 0:
            if isinstance(rem_ops[0], dict) and "machine_options" in rem_ops[0]:
                can_simulate = True
    
    if can_simulate:
        # Use a beam search with LPT-weighted scoring and smooth makespan proxy
        def smooth_max(values, beta=5.0):
            """Smooth max via log-sum-exp."""
            if not values:
                return 0.0
            v = np.array(values, dtype=float)
            v_max = np.max(v)
            if v_max < 0:
                v_max = 0.0
            shifted = v - v_max
            log_sum = np.log(np.sum(np.exp(shifted)))
            return v_max + np.log(len(v)) / beta + np.log(np.mean(np.exp(shifted - np.log(len(v))))) / beta
        
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
                        next_op = dict(rem[0])
                        next_op["remaining_work"] = max(0, op["remaining_work"] - min_dur)
                        if len(rem) > 1:
                            next_op["remaining_ops"] = rem[1:]
                        else:
                            next_op["remaining_ops"] = []
                        new_ready.append(next_op)
                else:
                    new_ready.append(other)
            return new_ready, comp_t
        
        def estimate_smooth_makespan(ready_ops, current_time):
            """Estimate makespan using smooth max over per-job completion estimates."""
            if not ready_ops:
                return current_time
            
            # Group by job to get per-job remaining work
            job_work = {}
            for op in ready_ops:
                jid = op["job"]
                job_work[jid] = job_work.get(jid, 0) + op["remaining_work"]
            
            if not job_work:
                return current_time
            
            # Per-job completion estimate: current_time + remaining_work / parallelism
            # In open shop, each job's ops can run in parallel on different machines
            job_completions = []
            for jid, work in job_work.items():
                # Estimate: at least work / (number of machines) but at least current_time
                job_completions.append(current_time + work * 0.8)
            
            # Smooth max of job completions
            return float(smooth_max(job_completions, beta=8.0))
        
        # Beam search: depth 2, width 4
        beam_width = 4
        depth = 2
        
        # Level 1: score all candidates with LPT-weighted + smooth makespan
        candidates = []
        for i in range(n):
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, i)
            makespan_est = estimate_smooth_makespan(new_ready, new_t)
            
            # LPT bonus: prefer operations with larger min_duration
            lpt_bonus = features[i]["min_dur"] * 0.3
            
            # Remaining ops penalty: jobs with more remaining ops benefit from starting now
            rem_ops_bonus = features[i]["n_rem_ops"] * 0.5
            
            # Combined score (lower is better)
            score = makespan_est - lpt_bonus - rem_ops_bonus
            candidates.append((score, i))
        
        if not candidates:
            # Fallback: pure LPT
            return int(np.argmax([f["min_dur"] for f in features]))
        
        candidates.sort(key=lambda x: x[0])
        top_k = candidates[:beam_width]
        
        # Level 2: for each top-1 candidate, explore next step
        best_score = float('inf')
        best_idx = top_k[0][1]
        
        for level1_score, level1_idx in top_k:
            if time_module.time() > deadline:
                break
            new_ready, new_t = simulate_one_step(ready_operations, time, level1_idx)
            
            if not new_ready:
                total_score = new_t
                if total_score < best_score:
                    best_score = total_score
                    best_idx = level1_idx
                continue
            
            for j in range(len(new_ready)):
                if time_module.time() > deadline:
                    break
                new_ready2, new_t2 = simulate_one_step(new_ready, new_t, j)
                makespan_est2 = estimate_smooth_makespan(new_ready2, new_t2)
                # Combine: use smooth max of level1 and level2 estimates
                total_score = float(smooth_max([level1_score, makespan_est2], beta=6.0))
                
                if total_score < best_score:
                    best_score = total_score
                    best_idx = level1_idx
        
        return best_idx
    
    # --- Fallback: LPT-weighted heuristic with smooth scoring ---
    # When simulation isn't available, use a weighted score emphasizing LPT
    
    # Compute per-job remaining work
    job_work = {}
    job_min_durs = {}
    for i, op in enumerate(ready_operations):
        jid = op["job"]
        job_work[jid] = job_work.get(jid, 0) + op["remaining_work"]
        if jid not in job_min_durs or op["machine_options"]:
            min_d = min(op["machine_options"].values()) if op["machine_options"] else 0
            job_min_durs[jid] = job_min_durs.get(jid, 0) + min_d
    
    # Smooth max of job works (proxy for makespan)
    job_works_list = list(job_work.values())
    if job_works_list:
        smooth_job_max = float(smooth_max(job_works_list, beta=10.0))
    else:
        smooth_job_max = 0.0
    
    scores = []
    for i, f in enumerate(features):
        # LPT: prefer long min_duration
        lpt_component = f["min_dur"] * 1.0
        
        # Remaining work of the job (to prioritize critical jobs)
        jid = ready_operations[i]["job"]
        job_rw = job_work.get(jid, 0)
        job_component = job_rw * 0.5
        
        # Flexibility: prefer less flexible ops (fewer machine options) to lock in early
        flex_component = (5 - f["n_machines"]) * 0.3
        
        # Number of remaining ops: jobs with more ops should start early
        rem_component = f["n_rem_ops"] * 0.4
        
        # Smooth makespan reduction: how much does starting this op reduce the smooth max?
        # Approximate: if this op is on the critical job, starting it helps most
        criticality = 1.0 / (f["remaining_work"] + 1.0) if f["remaining_work"] > 0 else 0.0
        
        score = lpt_component + job_component + flex_component + rem_component + criticality * 2.0
        scores.append(score)
    
    return int(np.argmax(scores))
