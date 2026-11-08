import time as _time
import numpy as np

# Deadline guard: ensure we return well before 150s wall-clock limit
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return 0
    
    best_idx = 0
    best_machine = None
    best_score = float('inf')
    
    n = len(ready_operations)
    
    # Precompute some static info for speed
    # We assume 'time' is the current global simulation time.
    # We don't have direct access to machine busy times in the standard interface,
    # but we can approximate the "urgency" by looking at the remaining work
    # and the specific machine constraints.
    
    # Strategy: Lookahead / Critical Path based
    # For each candidate, estimate the "criticality".
    # A simple lookahead: if we start op i on machine m, when does it finish?
    # We don't know m's busy time. However, we can use a heuristic that
    # prioritizes operations on machines that are likely bottlenecks or
    # operations that unlock long chains of work.
    
    # Let's try a different angle: "Most Critical Job" (MCJ) with tie-breaking by "Longest Processing Time" (LPT)
    # Or "Shortest Remaining Processing Time" (SRPT) but on the job level?
    
    # Let's implement a 1-step lookahead simulation assuming machines are free
    # (or using a penalty if we can't determine availability).
    # Actually, a more robust heuristic for Job Shop is "Critical Path" based.
    # Since we don't have the full graph, we approximate using remaining_work.
    
    # New Strategy: "Weighted Shortest Processing Time" (WSPT) combined with "Most Operations Remaining" (MOR)
    # The current program uses WSPT (remaining_work / min_dur).
    # Let's try "Critical Path Approximation":
    # Score = (remaining_work * 1.0) + (remaining_ops * 10.0) - (min_dur * 0.5)
    # This prioritizes jobs with more total work (critical path length proxy)
    # and more remaining steps (urgency), while slightly penalizing long current ops
    # to keep the pipeline moving.
    
    # Let's refine this to be more adaptive.
    # If we pick an op, we want to minimize the time until the job is done.
    
    # Let's try a "Lookahead" that simulates the next event.
    # For each candidate i:
    #   1. Pick best machine m (min duration).
    #   2. Estimate finish time = time + duration (assuming machine free).
    #   3. Calculate new remaining_work for job j.
    #   4. Score based on how much "critical work" is reduced.
    
    # Since we don't know machine availability, assuming it's free is optimistic.
    # If a machine is busy, the start is delayed.
    # Without machine state, we must rely on static heuristics derived from op properties.
    
    # Let's try "Most Remaining Operations" (MRO) as the primary key,
    # because jobs with more steps left are more likely to be on the critical path
    # and blocking other machines later.
    # Tie-breaker: Smallest min_duration (to free up the machine quickly).
    
    # Wait, the inspiration said "Multi-criteria heuristic combining WSPT ratio, critical path weight, and urgency bonus" improved score.
    # Let's implement a distinct variant: "Earliest Due Date" approximation using remaining work as due date proxy?
    # Or "Highest Priority" based on (remaining_ops / min_dur)?
    
    # Let's try: "Job Criticality" = remaining_work / remaining_ops (avg remaining op length)
    # We want to process jobs with HIGH criticality (long avg ops) first?
    # Or LOW criticality (short avg ops) to finish them?
    # Usually, WSPT (total work / processing time) is good for flow time.
    # For makespan, we want to keep machines busy and avoid bottlenecks.
    
    # Let's try a "Greedy Critical Path" approach:
    # 1. For each op, calculate min_dur.
    # 2. Calculate "Job Urgency" = remaining_ops (more steps = more future decisions).
    # 3. Calculate "Op Urgency" = 1 / min_dur (finish this op soon).
    # 4. Score = Job_Urgency * 0.5 + Op_Urgency * 0.5
    
    # Let's test a different combination:
    # Score = (remaining_work / min_dur) ** 0.5 + remaining_ops * 0.1
    
    # Actually, let's look at the "Inspiration 1" which scored 1.0616.
    # It used numpy.
    
    # Let's try a "Lookahead" that considers the *next* operation in the job.
    # If starting this op reveals a very long next op on a busy machine, maybe we should delay?
    # We don't have access to the next op's details easily (only remaining_ops and remaining_work).
    
    # Let's stick to a robust multi-factor heuristic that differs from the simple WSPT.
    # Factor 1: WSPT (remaining_work / min_dur)
    # Factor 2: Step Urgency (remaining_ops)
    # Factor 3: Machine Load Proxy (1/min_dur)
    
    # Let's try to maximize:
    # Score = (remaining_work / min_dur) + (remaining_ops * 0.5)
    
    # To be materially different, let's use a "Simulated Annealing" like randomization?
    # No, deterministic is better for reproducibility and speed.
    
    # Let's try "Most Remaining Work" (MRW) with a tie-breaker of "Fewest Remaining Ops".
    # This prioritizes heavy jobs.
    
    # Let's try "Shortest Remaining Processing Time" (SRPT) on the job level?
    # i.e., pick the job with the smallest total remaining_work.
    # This tends to clear small jobs out of the way, potentially reducing blocking.
    
    # Let's implement: "Smallest Remaining Work" (SRW)
    # Score = -remaining_work (we want to minimize this, so we maximize -remaining_work)
    # Tie-breaker: Smallest min_dur
    
    # Let's compare SRW vs WSPT.
    # WSPT favors high work/short time.
    # SRW favors low total work.
    
    # Let's try a hybrid: "Critical Ratio" = remaining_work / time_left? No time_left.
    
    # Let's try: "Most Operations Remaining" (MOR)
    # Jobs with more ops left are more complex and likely to be bottlenecks.
    # Score = remaining_ops
    # Tie-breaker: Smallest min_dur
    
    # Let's go with a "Weighted Critical Path" approximation:
    # We assume the critical path is proportional to the remaining work of the job.
    # But we also want to keep the machine utilization high.
    
    # Let's try:
    # Score = (remaining_work / min_dur) * 0.8 + (remaining_ops * 100.0) * 0.2
    # This balances WSPT with step urgency.
    
    # To be truly different, let's use a "Lookahead" that simulates the completion of the current op
    # and checks if it unlocks a "critical" next step.
    # Since we don't have the next step's machine, we just use remaining_ops.
    
    # Let's implement "Most Remaining Operations" (MRO) as the primary driver,
    # as jobs with more steps remaining have a higher probability of being on the critical path
    # and causing delays later if not started.
    
    for i, op in enumerate(ready_operations):
        options = op['machine_options']
        if not options:
            continue
        
        min_dur = min(options.values())
        best_machine_for_op = next(m for m, d in options.items() if d == min_dur)
        
        remaining_work = op['remaining_work']
        remaining_ops = op['remaining_ops']
        
        # Heuristic: Prioritize jobs with more remaining operations (MRO)
        # This targets the "depth" of the job, aiming to clear long chains early.
        # Tie-breaker: Smallest processing time (SPT) to reduce machine occupation.
        
        # Score = remaining_ops * 100.0 - min_dur
        # This strongly favors jobs with more steps left.
        
        # Alternative: "Weighted MRO"
        # Score = remaining_ops * (1.0 / min_dur)
        # This favors jobs with many steps AND short current ops.
        
        # Let's try: Score = remaining_ops * 1.0 + (1.0 / (min_dur + 1.0))
        
        # Let's try a different family: "Earliest Completion Time" (ECT)
        # Assume machine is free. Completion = time + min_dur.
        # We want to minimize the max completion time? No, that's makespan.
        # Minimizing the completion time of the *current* op might help if it unlocks the next.
        
        # Let's use "Shortest Processing Time" (SPT) on the operation,
        # but weighted by the job's remaining work.
        # Score = remaining_work / min_dur  (This is WSPT, already used)
        
        # Let's try "Most Remaining Work" (MRW)
        # Score = remaining_work
        # This prioritizes heavy jobs.
        
        # Let's try "Least Remaining Work" (LRW)
        # Score = -remaining_work
        
        # Let's try a "Hybrid" that is distinct from WSPT:
        # Score = (remaining_ops * 10.0) + (1.0 / min_dur)
        # This prioritizes depth and speed.
        
        score = (remaining_ops * 10.0) + (1.0 / (min_dur + 1.0))
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op

    return (best_idx, best_machine)
