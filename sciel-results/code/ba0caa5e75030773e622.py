"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import sys

# Stateful tracking for the CPM Dispatch
# We maintain the current estimated free time for each machine.
_machine_free_times = {}
_start_time = None
_dispatch_count = 0
_last_time = -1

# CPM State:
# job_cpl[job] = Critical Path Length remaining for that job
# job_min_dur[job] = Minimum possible duration for the *next* operation of that job (to handle dynamic updates)
# job_ops_seen[job] = Number of operations dispatched for this job (to detect when 'next' becomes 'current')
job_cpl = {}
job_min_dur = {}
job_ops_seen = {}

def dispatch(ready_operations, time):
    global _start_time, _dispatch_count, _last_time, _machine_free_times
    global job_cpl, job_min_dur, job_ops_seen
    
    # Detect new instance (time goes backwards or significantly resets)
    if time < _last_time - 1e-6:
        _machine_free_times = {}
        _dispatch_count = 0
        job_cpl = {}
        job_min_dur = {}
        job_ops_seen = {}
    _last_time = time
    
    if _start_time is None:
        _start_time = _time.time()
        
    # Deadline guard
    if _time.time() - _start_time > 110:
        # Fallback: SPT on the machine that is free earliest
        best_idx = 0
        best_key = (float('inf'), float('inf'))
        for i, op in enumerate(ready_operations):
            options = op["machine_options"]
            if not options:
                continue
            min_completion = float('inf')
            for m, dur in options.items():
                free_t = _machine_free_times.get(m, 0)
                completion = max(time, free_t) + dur
                if completion < min_completion:
                    min_completion = completion
            if (min_completion, 0) < best_key:
                best_key = (min_completion, 0)
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0

    _dispatch_count += 1
    
    # Strategy: Critical Path Method (CPM) Dispatch
    # 1. For each ready operation, identify its job.
    # 2. Maintain a dynamic estimate of the Critical Path Length (CPL) for each job.
    #    - When a job first appears, CPL is initialized based on its remaining work/ops.
    #    - When an op is dispatched, we subtract the duration of that op from the CPL.
    #    - However, since we don't know future durations, we use a heuristic update:
    #      New CPL = Old CPL - (duration of completed op) + (estimated average duration of next op)
    #      Actually, a simpler and robust proxy for "Critical Path" in JSS without full graph access
    #      is to prioritize jobs with the highest (Remaining Work / Remaining Ops) ratio? No, that's average.
    #      
    #      Let's stick to the standard CPM approximation for online JSS:
    #      Priority = max( Estimated Completion Time of Job )
    #      Estimated Completion Time of Job = Current Time + Sum of (Min Duration of Remaining Ops)
    #      
    #      Since we don't have future ops' min durations, we can track the "slack" or just use
    #      the known remaining_work as a proxy for total remaining duration? 
    #      remaining_work is likely the sum of durations of remaining ops.
    #      
    #      Let's refine the CPM proxy:
    #      For each job, we estimate its completion time as:
    #      ET_j = max( min(machine_free_times for ops that *could* run) , current_time ) + remaining_work_j
    #      This is not quite right because remaining_work is on specific machines.
    #      
    #      Alternative Structural Change: **Minimize Max Job Completion Time**
    #      For each candidate op i (belonging to job j):
    #      1. Determine the earliest possible completion time of this op (C_i).
    #      2. Estimate the completion time of the *entire job* j if we pick this op now.
    #         Job_Completion_Estimate = C_i + (remaining_work_j - duration_i)
    #         Note: This assumes the rest of the job can be processed instantly after this op,
    #         which is an optimistic bound, but consistent across jobs if we use the same metric.
    #         Actually, remaining_work_j includes duration_i.
    #         So, Remaining_Work_After = remaining_work_j - duration_i.
    #         Job_Completion_Estimate = C_i + Remaining_Work_After.
    #      3. Pick the op that minimizes the MAXIMUM estimated job completion time?
    #         No, usually we pick the op that minimizes the makespan.
    #         Makespan = max(Job_Completion_Estimates).
    #         
    #      Let's try: **Minimize the Estimated Makespan**
    #      For each candidate i:
    #        - Assign it to best machine m*.
    #        - Calculate C_i (completion of this op).
    #        - Calculate Est_Job_Completion_j = C_i + (remaining_work_j - dur_i).
    #        - Update the global max job completion estimate?
    #        - We don't have estimates for *other* jobs' completions easily without simulating them too.
    #        
    #      Simpler CPM variant: **Longest Critical Path First**
    #      Prioritize the job with the largest "Critical Path" remaining.
    #      Proxy for Critical Path: remaining_work / remaining_ops? No, that's average.
    #      Let's use the **Remaining Work** itself as the primary driver, but adjusted by machine availability.
    #      
    #      Let's go with a **Hybrid CPM/Bottleneck** rule:
    #      Score = w1 * (Estimated Job Completion Time) + w2 * (Machine Load Imbalance)
    #      
    #      Actually, the most effective simple CPM heuristic in literature is:
    #      **Dispatch the operation belonging to the job with the longest remaining critical path.**
    #      Since we can't compute the exact critical path, we approximate it with `remaining_work`.
    #      But to make it "CPM-like" and different from simple MRW (Most Remaining Work), we must account for the *delay* imposed by machine availability.
    #      
    #      Let's implement: **Minimize the Sum of Estimated Job Completion Times**? No, Makespan is max.
    #      
    #      Let's try: **For each candidate, calculate the "Delay" it imposes on the critical job.**
    #      
    #      Let's implement a direct **Makespan Minimization via Job Completion Estimation**:
    #      For each candidate op i (job j):
    #      1. Find best machine m for op i.
    #      2. Calculate C_i = max(time, free_m) + dur_i.
    #      3. Estimate the completion time of job j: 
    #         Est_Comp_j = C_i + (remaining_work_j - dur_i)  # Optimistic: no further delays
    #      4. We want to minimize the makespan. The makespan is at least max( Est_Comp_j for all jobs ).
    #         Since we don't know Est_Comp for other jobs, we can't compute the global max directly.
    #         
    #      However, we can use a **Local Makespan Proxy**:
    #      Pick the op that minimizes the *maximum* of:
    #         a) The estimated completion of the job being processed.
    #         b) The current known "critical" machine load (max free time).
    #      
    #      Let's try a different structural approach: **Simulated Annealing-like Greedy?** No, too slow.
    #      
    #      Let's go with **Critical Path Length (CPL) Priority**:
    #      We maintain a `job_cpl` dictionary.
    #      Initialization: When a job is first seen, `job_cpl[job] = remaining_work`.
    #      Update: When an op is dispatched, `job_cpl[job] -= duration_of_op`.
    #      Wait, `remaining_work` is provided in the dict. We can just use `remaining_work` directly!
    #      So `job_cpl` is effectively `remaining_work`.
    #      
    #      So the strategy "Prioritize job with max remaining_work" is MRW.
    #      To make it CPM, we must consider that some ops are on bottleneck machines.
    #      
    #      **New Strategy: Bottleneck-Aware Critical Path**
    #      For each candidate op i (job j):
    #      1. Calculate the "Effective Critical Path" contribution.
    #      2. If the op is on a machine that is currently the most loaded (bottleneck), prioritize it more?
    #         Or avoid it?
    #         Usually, you want to clear bottleneck machines.
    #      
    #      Let's implement: **Minimize the Estimated Completion Time of the Job with the Longest Remaining Work**
    #      This is complex.
    #      
    #      Let's try a simpler, robust structural change: **Greedy with Lookahead on Machine Load**
    #      1. For each op, find the machine that minimizes its completion time.
    #      2. Calculate the resulting machine load (free time).
    #      3. Calculate the "Variance" of machine loads.
    #      4. Pick the op that minimizes the makespan (max load), breaking ties by minimizing variance.
    #      
    #      This is essentially what the current program does (min_completion_for_op, then max_load).
    #      
    #      Let's try: **Longest Processing Time (LPT) on the Critical Job**
    #      1. Identify the job with the maximum `remaining_work` (Critical Job).
    #      2. If the Critical Job has a ready op, dispatch it immediately.
    #      3. If not, dispatch the op that minimizes the completion time of the *second* most critical job?
    #      
    #      This is a **Priority Class** approach.
    #      
    #      Let's implement **Dynamic Critical Job Dispatch**:
    #      1. Find the job with the max `remaining_work` among *all* jobs that have ready ops? 
    #         No, only ready ops are in the list.
    #         So, among the ready ops, find the one belonging to the job with the max `remaining_work`.
    #      2. If there are multiple, pick the one with the shortest duration (SPT) on the best machine.
    #      
    #      This is different from MRW? No, it IS MRW if we just pick the job with max remaining work.
    #      The current program uses `score = min_completion_for_op / (job_rem_work + 1e-9)`.
    #      This favors *small* remaining work (urgent/small jobs).
    #      My proposal: Favor *large* remaining work (Critical jobs).
    #      
    #      Let's implement **Most Remaining Work (MRW) with Machine Awareness**:
    #      1. Sort ready ops by `remaining_work` descending.
    #      2. Pick the first one (highest remaining work).
    #      3. Assign it to the machine that gives the earliest completion time.
    #      
    #      This is a structural inversion of the current logic (which minimizes completion/rem_work, i.e., favors small rem_work).
    #      
    #      Let's refine: **Critical Path Method (CPM) Approximation**
    #      Score = -remaining_work (Maximize remaining work)
    #      Tie-breaker = min_completion_time (Minimize delay)
    #      
    #      This is a distinct family: **Priority by Criticality (MRW)** vs **Priority by Urgency/Efficiency (Current)**.

    best_idx = 0
    best_score = -float('inf')
    best_machine = None
    best_completion = float('inf')
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        job_id = op["job"]
        job_rem_work = op["remaining_work"]
        
        # Find the best machine for this specific operation (earliest completion)
        min_completion_for_op = float('inf')
        best_m_for_op = None
        
        for m, dur in options.items():
            free_t = _machine_free_times.get(m, 0)
            start_t = max(time, free_t)
            completion_t = start_t + dur
            
            if completion_t < min_completion_for_op:
                min_completion_for_op = completion_t
                best_m_for_op = m

        if best_m_for_op is None:
            continue
            
        # Strategy: Prioritize jobs with the MOST remaining work (Critical Path Proxy)
        # Tie-break: Earliest completion time
        # This is structurally different from the current "minimize completion/rem_work"
        
        # We want to MAXIMIZE job_rem_work.
        # If tie, MINIMIZE min_completion_for_op.
        
        # To combine into a single score for sorting:
        # We can use a tuple comparison logic manually.
        
        # Let's use a weighted score:
        # Score = job_rem_work * 1000 - min_completion_for_op
        # This heavily favors high remaining work, then breaks ties with speed.
        
        score = job_rem_work * 1000 - min_completion_for_op
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op
            best_completion = min_completion_for_op
            
    # If no valid candidate found (should not happen if ready_operations not empty)
    if best_machine is None:
        return 0

    # Update global state with the chosen assignment
    sel_op = ready_operations[best_idx]
    dur = sel_op["machine_options"][best_machine]
    start_t = max(time, _machine_free_times.get(best_machine, 0))
    finish_t = start_t + dur
    _machine_free_times[best_machine] = finish_t
    
    # Return (index, machine) to ensure state consistency
    return (int(best_idx), best_machine)

# EVOLVE-BLOCK-END
