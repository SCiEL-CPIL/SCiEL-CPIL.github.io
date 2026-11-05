"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track machine availability if needed, but we rely on local simulation
# We need to be careful not to mutate the passed-in state.

def _estimate_makespan(ready_operations, time, chosen_index, chosen_machine=None):
    """
    Simulate the future schedule assuming ready_operations[chosen_index] is started now.
    We use a simplified greedy simulation for the rest of the jobs.
    """
    # Copy state to avoid side effects
    # ready_operations is a list of dicts. We need to simulate all jobs, not just ready ones?
    # The problem says "ready_operations" are those ready to start.
    # We don't have access to the full state of non-ready operations (those waiting for predecessors).
    # However, the heuristic must work with only what is given.
    # If we only simulate ready operations, we are ignoring the pipeline of future operations.
    # A better proxy: Estimate the "urgency" by looking at the total remaining work and machine load.
    
    # Let's try a different structural change: 
    # Instead of simulating the whole future (which we can't fully do without global state),
    # we use a "Regret" based approach or a "Critical Path" approximation.
    
    # Actually, let's look at the inputs again.
    # ready_operations: list of dicts with job, op, machine_options, remaining_ops, remaining_work.
    # We can estimate the critical path length for each job.
    
    # Strategy: Select the operation that minimizes the estimated completion time of its job,
    # while balancing machine load.
    
    # Let's implement a "Min-Regret" or "Dynamic Priority" based on:
    # 1. Critical Path Length (CPL) of the job: remaining_work / parallelism? 
    #    We don't know parallelism. But remaining_work is a good proxy for total time.
    # 2. Machine Congestion: For the chosen operation, how long will the machine be busy?
    #    We don't have machine busy times in the input.
    
    # Wait, the prompt says: "optionally, which eligible machine runs it".
    # If we return (index, machine), we can influence the machine assignment.
    
    # Let's try a "Lookahead" on the *immediate* impact.
    # For each ready operation i:
    #   For each machine m in i.machine_options:
    #     duration = i.machine_options[m]
    #     # Heuristic: Prefer machines that are less loaded? We don't know load.
    #     # Heuristic: Prefer shortest duration (SPT) to free up machine quickly?
    #     # Heuristic: Prefer longest duration? No, usually SPT is good for makespan in JSS?
    #     # Actually, in FJSP, SPT on the critical job is often good.
    
    # Let's try a "Weighted Shortest Processing Time" combined with "Remaining Work".
    # Priority = (remaining_work * alpha) + (min_duration * beta)
    # But this is just tuning the current approach.
    
    # STRUCTURAL CHANGE: 
    # Use a "Greedy Pairing" approach. 
    # Identify the job with the longest remaining work (Critical Job).
    # Among the ready operations of that job, pick the one that can be done fastest on any machine.
    # If multiple jobs have similar remaining work, break ties by shortest processing time.
    
    # This focuses on the critical path directly, rather than just global remaining work.
    
    pass

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Find the job with the maximum remaining work (Critical Job)
    # If there are multiple ready ops for the same job, we pick the best one for that job.
    # If there are ready ops for different jobs, we prioritize the job with more remaining work.
    
    # Step 1: Group ready operations by job
    jobs = {}
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        if job_id not in jobs:
            jobs[job_id] = []
        jobs[job_id].append((i, op))
    
    # Step 2: Rank jobs by remaining work (descending)
    # We assume remaining_work is consistent for the job or we take the max from ready ops?
    # "remaining_work" is per operation in the dict? "remaining_work" usually means total work left for the job.
    # Let's assume op["remaining_work"] is the total remaining work for that job.
    
    job_priorities = []
    for job_id, ops in jobs.items():
        # Use the remaining_work of the first op (should be same for all ops of the job)
        rem_work = ops[0][1]["remaining_work"]
        job_priorities.append((rem_work, job_id, ops))
        
    # Sort by remaining work descending
    job_priorities.sort(key=lambda x: x[0], reverse=True)
    
    # Step 3: For the top priority job, select the best operation
    # If the top job has multiple ready ops, pick the one with the shortest min-duration (SPT)
    # to minimize the time this critical job spends on the machine.
    
    top_job = job_priorities[0]
    top_job_id = top_job[1]
    top_ops = top_job[2]
    
    # If there's only one job with ready ops, we still need to pick the best op for it.
    # If there are multiple jobs, we picked the job with most remaining work.
    # Now, among the ready ops of this job, which one to start?
    # Usually, there's only one ready op per job at a time in JSS, but let's handle multiple.
    
    best_idx = None
    best_score = float('inf')
    
    for idx, op in top_ops:
        # Score: Minimize the minimum duration among available machines (SPT)
        min_dur = min(op["machine_options"].values())
        # Tie-breaker: If same min_dur, pick the one with fewer options? Or just first.
        if min_dur < best_score:
            best_score = min_dur
            best_idx = idx
            
    # Return the index
    return int(best_idx)

# EVOLVE-BLOCK-END
