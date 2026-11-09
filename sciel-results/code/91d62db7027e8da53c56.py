"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import heapq

# Deadline guard
_START_TIME = None
_DEADLINE = 145.0

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Deadline check
    elapsed = _time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        # Fallback: MWR (Most Work Remaining)
        if not ready_operations:
            return 0
        return int(max(range(len(ready_operations)), key=lambda i: ready_operations[i].get("remaining_work", 0)))
    
    try:
        if not ready_operations:
            return 0
        
        n = len(ready_operations)
        
        # If there's only one option, pick it
        if n == 1:
            return 0
        
        # Pre-compute static features for quick filtering
        # 1. Remaining Work (RW)
        # 2. Longest Remaining Path (LRP) approximation: sum of min durations for remaining ops
        # 3. Min Duration of current op
        
        static_features = []
        for op in ready_operations:
            rw = op.get("remaining_work", 0)
            mo = op.get("machine_options", {})
            min_dur = min(mo.values()) if mo else 1.0
            
            # Estimate LRP: current min_dur + sum of min durations of remaining ops?
            # We don't have the remaining ops' durations directly, only count.
            # However, 'remaining_work' is likely the sum of durations.
            # Let's assume remaining_work is a good proxy for the total work left.
            # A better proxy for "criticality" is often the longest path through the remaining DAG.
            # Since we only have aggregate 'remaining_work', we can use it as a proxy for length.
            
            static_features.append({
                'rw': rw,
                'min_dur': min_dur,
                'mo': mo
            })
        
        # Strategy: Lookahead Simulation
        # We simulate the dispatch of the current set of ready operations.
        # For each candidate i in ready_operations:
        #   1. Assign candidate i to its best machine (min duration) at current 'time'.
        #   2. For all other ready operations, simulate a greedy dispatch (e.g., SPT or MWR) 
        #      on a virtual timeline starting from 'time'.
        #   3. Estimate the "makespan" or "critical completion time" of this simulated batch.
        #   4. Choose the candidate i that leads to the lowest estimated completion time 
        #      for the most critical job, or the lowest overall makespan estimate.
        
        # To keep it fast, we limit the lookahead depth or simplify the simulation.
        # A simple "greedy batch completion time" estimator:
        # If we pick op i, it finishes at time + min_dur_i.
        # The other ops will be processed. The bottleneck is usually the job with the most remaining work.
        
        # Let's try a "Critical Path" based heuristic:
        # Priority = Longest Remaining Path (LRP)
        # LRP_j = max over all paths in job j's remaining DAG of sum(durations).
        # Approximation: remaining_work is sum of durations. LRP is usually <= remaining_work.
        # In flexible job shop, the critical job is the one with the highest LRP.
        
        # Let's refine the score:
        # Score = LRP_proxy / (1 + congestion_estimate)
        
        # Congestion Estimate:
        # How busy are the machines? We don't have global machine state in the input, 
        # only 'time' and 'ready_operations'. 
        # However, we can infer congestion if we assume the simulation maintains some state? 
        # No, the interface is stateless per call except for 'time'.
        # Wait, 'ready_operations' contains 'machine_options'. 
        # If we can't see machine availability, we must assume the simulator handles machine assignment 
        # or we must return (index, machine).
        
        # The prompt says: "optionally (index, machine)".
        # If we return just index, the simulator picks a machine? 
        # "picks which ready operation to start next and, optionally, which eligible machine runs it"
        # If we return just index, the simulator likely picks the machine that minimizes completion time 
        # or uses a default rule. 
        # To maximize control, we should return (index, machine) if possible, 
        # but the fallback in the current program returns just int(index).
        # Let's check if returning a tuple is safe. The prompt says "optionally".
        # If I return a tuple, I need to be sure the evaluator accepts it.
        # "Interface: dispatch(ready_operations, time) -> index ... (optionally (index, machine))"
        # It's safer to return just the index if I'm not sure about the machine selection logic 
        # interacting with the simulator's internal state, UNLESS I can simulate it perfectly.
        
        # Let's stick to returning the index, but use a smarter index selection.
        
        # New Strategy: "Longest Processing Time (LPT) on Critical Jobs"
        # 1. Identify the job with the highest 'remaining_work' (proxy for criticality).
        # 2. Among ready ops of that job, pick the one with the longest min duration? 
        #    No, usually you want to clear the critical path. 
        #    If the critical job has multiple ready ops, pick the one that unlocks the most subsequent work? 
        #    We don't have graph structure.
        
        # Let's try a "Shortest Processing Time (SPT) for non-critical, Longest for critical" hybrid.
        
        # Actually, the "lookahead" idea was good. Let's implement a lightweight simulation.
        # Simulate the completion of the current batch of ready operations.
        # Assume all machines are free at 'time' (worst case for congestion, best case for throughput).
        # If we assume machines are free, the makespan of the batch is determined by the longest job in the batch 
        # if we can parallelize on different machines.
        
        # Let's group ready operations by job.
        # jobs_in_batch = {}
        # for i, op in enumerate(ready_operations):
        #     j = op['job']
        #     if j not in jobs_in_batch:
        #         jobs_in_batch[j] = []
        #     jobs_in_batch[j].append(i)
        
        # For each job in the batch, the time it takes to finish its ready ops is the sum of durations 
        # if they are sequential (which they are in a job).
        # So, for job j, batch_time_j = sum(min_dur of each ready op in job j).
        
        # If we pick op i (from job j_i), it starts now.
        # The other ops from job j_i must wait for op i to finish? 
        # No, "ready_operations" are all ready to start. 
        # In a job shop, a job has a sequence. Only one op of a job is ready at a time usually?
        # "ready_operations" implies these are all currently available.
        # If a job has multiple ready ops, they might be parallelizable? 
        # Standard job shop: one op at a time per job.
        # So if a job has multiple entries in ready_operations, it's non-standard or they are parallel?
        # Let's assume standard: each job has at most 1 ready op.
        # If so, batch_time for each job is just the min_dur of its ready op.
        
        # If each job has 1 ready op, then we can run all of them in parallel on different machines.
        # The makespan of this batch is max(min_dur_i).
        # This doesn't help differentiate much unless we consider future work.
        
        # Let's look at "remaining_work".
        # Job with highest remaining_work is the bottleneck.
        # We should prioritize the op from the job with highest remaining_work.
        # If multiple ops from the same job are ready (unlikely), pick the longest?
        
        # Let's try: 
        # 1. Group by job.
        # 2. For each job, calculate "urgency" = remaining_work / (1 + number_of_ready_ops_in_job).
        # 3. Pick the job with highest urgency.
        # 4. Among ops of that job, pick the one with min duration (to finish it fast and unlock next).
        
        # This is essentially MWR (Most Work Remaining) with a twist.
        # The current program uses 0.7 * work + 0.3 * efficiency.
        # MWR is often strong. Let's see if we can improve the "efficiency" part.
        
        # Alternative: "Critical Ratio"
        # CR = (Time to due date) / (Work remaining). No due dates.
        
        # Let's go back to the simulation idea but simpler.
        # Estimate the "completion time" of the entire system if we pick i.
        # This is hard without global state.
        
        # Let's try a different heuristic: "Minimum Slack"
        # Slack = (Latest start time) - (Earliest start time). No due dates.
        
        # Let's try "Shortest Remaining Processing Time (SRPT)" for the job with most remaining work.
        
        # Let's try a composite score that penalizes long durations more heavily 
        # for jobs that are already "long" (high remaining work).
        # Score = remaining_work - alpha * min_duration
        # If remaining_work is high, we want to start it. 
        # But if the duration is very long, maybe we should start a shorter one to keep the machine busy?
        # No, in makespan minimization, blocking a machine with a long job is bad if that job is critical.
        # If the job is NOT critical, a short job is better.
        
        # Let's define "Criticality" = remaining_work / max_remaining_work.
        # If Criticality > 0.8, use MWR (pick highest remaining work).
        # If Criticality < 0.8, use SPT (pick shortest duration).
        
        # This is a "Hybrid" strategy.
        
        max_rw = max(f['rw'] for f in static_features)
        if max_rw == 0:
            return 0
            
        best_idx = 0
        best_score = -1e10
        
        for i, f in enumerate(static_features):
            rw = f['rw']
            min_dur = f['min_dur']
            
            # Normalize RW
            rw_norm = rw / max_rw
            
            # If this job is very critical (high RW), prioritize it strongly.
            # If not critical, prioritize short duration to keep flow high.
            
            # Score = w1 * rw_norm - w2 * (min_dur / max_min_dur)
            # Let's find max_min_dur for normalization
            # We can compute it on the fly or pre-compute.
            
            # Let's pre-compute max_min_dur
            # (Already have static_features)
            
            # Let's use a dynamic weight:
            # If rw_norm is high, weight RW more.
            # weight_rw = 0.5 + 0.5 * rw_norm
            # weight_dur = 1 - weight_rw
            
            # Score = weight_rw * rw_norm - weight_dur * (min_dur / max_possible_dur)
            # What is max_possible_dur? Let's use max_min_dur in the batch.
            
            # This is getting complex. Let's simplify.
            
            # Strategy: "Weighted Shortest Processing Time"
            # Score = (remaining_work + 1) / (min_duration + 1)
            # This favors high work and short duration.
            
            score = (rw + 1.0) / (min_dur + 1.0)
            
            # Tie breaker: higher remaining work
            if score > best_score + 1e-9:
                best_score = score
                best_idx = i
            elif abs(score - best_score) <= 1e-9:
                if rw > static_features[best_idx]['rw']:
                    best_idx = i
        
        return int(best_idx)

    except Exception:
        if not ready_operations:
            return 0
        return 0

# EVOLVE-BLOCK-END
