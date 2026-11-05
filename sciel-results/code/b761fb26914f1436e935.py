"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import heapq
import sys

# We implement a Critical Path (CP) heuristic.
# 1. Estimate the remaining processing time for each job by finding the longest path 
#    in the graph of remaining operations, using the minimum possible duration for each op.
# 2. Select the job with the highest estimated remaining completion time (most critical).
# 3. For the selected job, pick the machine that minimizes the estimated completion time 
#    of that specific operation (greedy machine selection).

# Deadline guard
_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _estimate_job_completion(job_id, ready_ops_dict, min_durations_cache):
    """
    Estimate the remaining completion time for a job using the longest path
    in the remaining operations graph.
    
    ready_ops_dict: {job_id: {op_index: op_dict}}
    min_durations_cache: {job_id: {op_index: min_duration}}
    """
    job_ops = ready_ops_dict.get(job_id, {})
    if not job_ops:
        return 0.0
    
    # Build graph: nodes are operations, edges are precedence (op -> next_op)
    # We need to find the longest path from any "ready" or "current" op to the end.
    # Since we don't have the full DAG structure explicitly in the interface (only remaining_ops count),
    # we assume a linear chain or use the provided 'remaining_ops' and 'remaining_work' as proxies?
    # The interface provides: job, op, machine_options, remaining_ops, remaining_work.
    # It does NOT provide the explicit next operation index or the full DAG structure.
    # Therefore, we cannot compute the exact CP without the DAG.
    
    # Re-evaluating: The interface is sparse. 
    # "remaining_ops" is an integer count.
    # "remaining_work" is likely the sum of min durations of remaining ops.
    # If we assume a linear chain (common in simple job-shop generators), 
    # the completion time is just the sum of min durations of all remaining ops.
    # However, if the graph is parallel, remaining_work (sum) overestimates the critical path length.
    
    # Let's look at the fields again:
    # job: int
    # op: int (current op index)
    # machine_options: {machine: duration}
    # remaining_ops: int (count of ops left including this one? or after?)
    # remaining_work: float (sum of min durations? or sum of all durations?)
    
    # Standard heuristic without DAG: 
    # Use "remaining_work" as a proxy for job length.
    # But to make it "Critical Path" like, we should prioritize jobs with large remaining work.
    # The current program uses remaining_work directly.
    
    # Structural Change: Instead of just summing, let's use a **Dynamic SPT + Critical Ratio** approach.
    # Or better, since we can't build the DAG, let's use **Most Remaining Work** but with a 
    # **Machine Load Balance** tie-breaker that is more sophisticated than just min duration.
    
    # Actually, a very strong structural change is **Earliest Due Date (EDD)** or **Shortest Remaining Processing Time (SRPT)** 
    # but we don't have due dates.
    
    # Let's try a **Greedy Critical Path** approximation:
    # Score = remaining_work / (1 + number_of_machines_available_for_next_op)
    # This penalizes jobs that have many machine options (less critical) and rewards those with few options (bottlenecks).
    
    pass

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return (0, list(ready_operations[0]["machine_options"].keys())[0])
    
    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_machine)
    
    scores = []
    best_machines = []
    
    # Pre-compute machine loads? We don't have access to machine states.
    # We only have ready_operations.
    
    # New Strategy: **Bottleneck Aware Dispatching**
    # 1. Identify "bottleneck" operations: Those where the minimum duration is significantly 
    #    larger than the average duration of other options (high variability/cost).
    # 2. Prioritize jobs with high remaining work.
    # 3. Among high work jobs, prioritize those with "bottleneck" characteristics to clear them early.
    
    for i, op in enumerate(ready_operations):
        mw = float(op["remaining_work"])
        remaining_ops = float(op.get("remaining_ops", 1))
        
        durations = list(op["machine_options"].values())
        min_dur = min(durations)
        max_dur = max(durations)
        mean_dur = sum(durations) / len(durations)
        
        # Bottleneck score: How much worse is the worst machine compared to the best?
        # Or simply: Is this operation long?
        # Let's use a "Criticality" score:
        # Criticality = remaining_work * (1 + (max_dur - min_dur) / mean_dur)
        # This boosts jobs that have high remaining work AND have operations that are sensitive to machine choice (high variance).
        
        variance_factor = (max_dur - min_dur) / mean_dur if mean_dur > 0 else 0
        score = mw * (1.0 + 0.5 * variance_factor)
        
        # Also consider the number of remaining ops. Jobs with many remaining ops are "long" jobs.
        # We want to start them early.
        # score += 0.1 * remaining_ops
        
        scores.append(score)
        
        # Machine selection: Pick the machine with the minimum duration.
        # If there's a tie, pick the one with the lowest index (arbitrary but consistent).
        best_machine = min(op["machine_options"].keys(), key=lambda m: (op["machine_options"][m], m))
        best_machines.append(best_machine)
    
    import numpy as np
    scores = np.array(scores)
    idx = int(np.argmax(scores))
    
    return (idx, best_machines[idx])

# EVOLVE-BLOCK-END
