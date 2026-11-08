import time
import numpy as np
import copy

# Global deadline guard
_DEADLINE = time.time() + 145.0

def dispatch(ready_operations, time):
    """
    Dispatches the next operation using a greedy lookahead simulation.
    
    Strategy:
    - Instead of using a static formula, we simulate the completion of all
      currently ready operations.
    - For each candidate operation (and its best machine), we "complete" it
      and update the state of the system (reducing remaining work/ops).
    - We estimate the final makespan as the maximum completion time across
      all jobs, assuming the remaining work of each job is processed
      sequentially after its current ready operations are done.
    - We select the candidate that yields the minimum estimated makespan.
    
    This approach captures interactions between jobs (e.g., finishing a job
    early frees it from the critical path) better than static heuristics.
    """
    if not ready_operations:
        return -1

    # If we are close to the deadline, fallback to a fast heuristic (SPT)
    if time.time() > _DEADLINE:
        best_i = 0
        best_dur = float('inf')
        best_m = -1
        for i, op in enumerate(ready_operations):
            min_d = min(op['machine_options'].values())
            if min_d < best_dur:
                best_dur = min_d
                best_i = i
                best_m = min(op['machine_options'], key=op['machine_options'].get)
        return (best_i, best_m)

    n_jobs = 0
    for op in ready_operations:
        if op['job'] > n_jobs:
            n_jobs = op['job']
    n_jobs += 1

    best_score = float('inf')
    best_choice = (0, -1)

    # We need to simulate the effect of picking each candidate.
    # To avoid O(N^2) full simulations for large N, we use a simplified
    # "completion time" model.
    
    # Pre-compute the "base" remaining work for each job
    # Note: ready_operations contains the current state.
    # If we pick op A, job A's remaining work decreases by dur_A.
    # Other jobs' remaining work stays the same (for the first step).
    # However, the "makespan" is driven by the job that finishes last.
    
    # Let's estimate the "current load" of each job.
    # load[job] = max completion time among its ready ops + remaining_work_after_ready
    
    # Actually, a simpler and robust lookahead:
    # For each candidate (i, m):
    # 1. Calculate completion time of this op: t_end = time + dur
    # 2. For the job of this op: new_remaining_work = rem_work - dur
    # 3. For all OTHER jobs: their state is unchanged.
    # 4. Estimate makespan:
    #    For each job j:
    #       If j is the job of the picked op:
    #          estimated_finish_j = t_end + (new_remaining_work / max_parallelism?)
    #          Since we don't know future machine availability, assume serial or
    #          use the remaining_work as a proxy for time.
    #          Let's assume remaining_work is a time estimate.
    #       Else:
    #          estimated_finish_j = current_max_ready_time_j + remaining_work_j
    
    # To make this efficient, we pre-compute the "contribution" of each job
    # to the makespan if we do nothing, then adjust for the picked op.
    
    # Job State:
    # max_ready_time[job] = max(time + dur) for all ready ops of that job?
    # No, in a real sim, only ONE op of a job runs at a time.
    # But here, multiple ops of the same job might be ready? 
    # "ready_operations" usually implies all ops that can start.
    # In JSS, a job has a sequence. So only ONE op per job should be ready 
    # unless it's a flexible graph. The dict has 'remaining_ops', implying a sequence.
    # So, likely at most 1 ready op per job.
    
    # Let's assume at most 1 ready op per job.
    # If multiple, we treat them as independent branches, but standard JSS is serial.
    # Let's assume 1 ready op per job.
    
    job_data = {}
    for i, op in enumerate(ready_operations):
        j = op['job']
        if j not in job_data:
            job_data[j] = {
                'index': i,
                'rem_work': op.get('remaining_work', 0),
                'min_dur': min(op['machine_options'].values()),
                'best_m': min(op['machine_options'], key=op['machine_options'].get)
            }
        # If multiple ready ops for same job, this logic gets complex.
        # For now, assume the first one is the "head" of the job's ready queue.
        # Or just take the one with min duration?
        # Let's stick to the assumption that the list contains the current active op per job.

    # Estimate baseline makespan (if we pick nothing, but we must pick one)
    # Actually, we want to pick the one that minimizes the MAX over all jobs of:
    # (time when job's current op finishes) + (remaining work after current op)
    
    # For a candidate (i, m):
    # dur = op[i]['machine_options'][m]
    # t_finish = time + dur
    # job = op[i]['job']
    # new_rem_work = job_data[job]['rem_work'] - dur  # Note: rem_work usually includes current op?
    # The dict says 'remaining_work'. In many sims, this includes the current op.
    # If it includes current op, then remaining_after = rem_work - dur.
    # If it excludes, remaining_after = rem_work.
    # Let's assume it INCLUDES the current op (standard for "remaining workload").
    
    # For other jobs k:
    # t_finish_k = time + job_data[k]['min_dur']  # Assume they start now?
    # No, they can't all start now if machines are shared.
    # This is the weakness of simple lookahead.
    
    # Better heuristic:
    # Score = max(
    #   time + dur_candidate + (rem_work_candidate - dur_candidate),  # Job i finishes
    #   max_{k != i} (time + min_dur_k + rem_work_k)                  # Others finish
    # )
    
    # This assumes others start immediately. This is optimistic but consistent.
    
    # Pre-compute "other_max" for each potential candidate?
    # other_max for candidate i is the max over all k != i of (time + min_dur_k + rem_work_k).
    
    # Let's compute val_k = time + min_dur_k + rem_work_k for all k.
    vals = []
    for j, data in job_data.items():
        v = time + data['min_dur'] + data['rem_work']
        vals.append((v, j))
    
    # Sort vals to easily find max excluding i
    # Since N is small (10 jobs), we can just loop.
    
    for i, op in enumerate(ready_operations):
        j = op['job']
        # Try all machines? Or just the best?
        # Choosing the machine with min duration is usually optimal for makespan
        # unless it causes a bottleneck. But we don't have machine load info.
        # Let's try the machine that gives min duration.
        
        # However, the interface allows returning (index, machine).
        # We should probably evaluate the best machine for this op.
        
        best_m_for_op = min(op['machine_options'], key=op['machine_options'].get)
        dur = op['machine_options'][best_m_for_op]
        
        # Calculate score for picking (i, best_m_for_op)
        # Job i's projected finish:
        # If rem_work includes current op: rem_after = rem_work - dur
        # Else: rem_after = rem_work
        # Let's assume rem_work includes current op.
        rem_work_i = job_data[j]['rem_work']
        rem_after_i = max(0, rem_work_i - dur)
        finish_i = time + dur + rem_after_i
        
        # Other jobs' projected finish (assuming they start now with min dur)
        # We need max over k != j of (time + min_dur_k + rem_work_k)
        # Let's pre-compute the global max and second max?
        # Or just loop since N is small.
        
        max_other = 0
        for k, data in job_data.items():
            if k == j:
                continue
            v = time + data['min_dur'] + data['rem_work']
            if v > max_other:
                max_other = v
        
        score = max(finish_i, max_other)
        
        if score < best_score:
            best_score = score
            best_choice = (i, best_m_for_op)

    return best_choice
