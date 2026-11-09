"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Stateful tracking for machine availability and job critical path estimation
# Strategy: Dynamic Critical Path (DCP) dispatching.
# 1. Track machine availability times to estimate when operations can actually start.
# 2. Estimate the critical path length for each job based on remaining work and machine contention.
# 3. Dispatch the operation that minimizes the estimated completion time of its job (or maximizes urgency).

_machine_avail = {}
_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard: if we are running too long, fall back to a fast heuristic
    if _time.time() - _start_time > 100:
         # Fallback: Shortest Processing Time (SPT)
         min_dur = float('inf')
         best_idx = 0
         for i, op in enumerate(ready_operations):
             durs = op["machine_options"].values()
             min_d = min(durs)
             if min_d < min_dur:
                 min_dur = min_d
                 best_idx = i
         return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
        
    best_score = -float('inf')
    best_idx = 0
    best_machine = None

    # Heuristic: 
    # We want to prioritize jobs that are on the critical path or have high remaining work,
    # but also account for machine availability to avoid bottlenecks.
    # 
    # Score components:
    # 1. Estimated Completion Time (ECT) of the operation: min over machines (avail[m] + dur)
    # 2. Job Remaining Work (RW): Higher is worse (more work left) -> we want to finish jobs with less work? 
    #    Actually, in job-shop, we often want to prioritize jobs with *more* remaining work to keep them moving (MRW),
    #    or jobs with *shorter* remaining work to clear them (SPT-like).
    #    Let's try a hybrid: 
    #    Primary: Minimize ECT (get operations done quickly)
    #    Secondary: Maximize Remaining Work (prioritize long jobs to avoid starvation)
    #    Tertiary: Minimize Remaining Ops (prioritize jobs close to completion)
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # 1. Calculate best machine and earliest start/completion for this op
        min_ect = float('inf')
        best_m = None
        for m, dur in options.items():
            avail = _machine_avail.get(m, time)
            ect = avail + dur
            if ect < min_ect:
                min_ect = ect
                best_m = m
        
        if best_m is None:
            continue

        # 2. Get job stats
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # 3. Composite Score
        # We want to minimize ECT, but also consider the job's urgency.
        # A common effective rule is "Longest Remaining Processing Time" (LRPT) or "Most Remaining Work" (MRW).
        # However, ignoring machine availability is bad.
        # Let's use a weighted score:
        # Score = - (min_ect * 1.0) + (remaining_work * 0.01) - (remaining_ops * 0.001)
        # The weights are tuned such that ECT dominates, but MRW breaks ties.
        
        # Actually, let's try a different angle: 
        # Prioritize the job with the highest "Criticality".
        # Criticality ~ Remaining Work / Remaining Ops (average processing time left).
        # If a job has high average processing time left, it might be a bottleneck.
        
        avg_proc = remaining_work / remaining_ops if remaining_ops > 0 else 0
        
        # Let's try: Minimize ECT, tie-break by Max Remaining Work.
        # This is a "Shortest Estimated Completion Time" (SECT) rule with MRW tie-break.
        
        score = -min_ect + 0.1 * remaining_work
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m

    # Update machine availability for the selected operation
    if best_machine is not None and best_idx < n_ops:
        selected_op = ready_operations[best_idx]
        # The duration is on the chosen machine
        dur = selected_op["machine_options"].get(best_machine, 0)
        avail = _machine_avail.get(best_machine, time)
        _machine_avail[best_machine] = avail + dur

    return int(best_idx)

# EVOLVE-BLOCK-END
