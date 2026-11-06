"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = None

def dispatch(ready_operations, time):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    
    # Safety guard: if we are approaching the 120s limit, fall back to a fast heuristic
    # to ensure we don't timeout. The deadline is 120s, we check against 115s.
    if _time.time() - _START_TIME > 115:
        # Fast fallback: pick the operation with the shortest duration on its preferred machine
        # to keep the system moving.
        min_dur = float('inf')
        best_idx = 0
        for i, op in enumerate(ready_operations):
            min_op_dur = min(op["machine_options"].values())
            if min_op_dur < min_dur:
                min_dur = min_op_dur
                best_idx = i
        return int(best_idx)

    n_ops = len(ready_operations)
    if n_ops == 0:
        return 0
    if n_ops == 1:
        return 0

    # Strategy: Lookahead Simulation / Earliest Completion Time
    # Instead of a static weighted score, we simulate the immediate consequence of dispatching
    # each operation. We assume the machine is free at 'time' (since the op is ready) and
    # calculate the finish time for the best possible machine for that op.
    # We prioritize the operation that finishes the earliest (Min Finish Time).
    # Tie-breaking:
    # 1. Higher remaining_work (urgency: long jobs should not be starved)
    # 2. Fewer remaining_ops (urgency: jobs close to completion)
    
    best_candidate = None
    best_key = None

    for i, op in enumerate(ready_operations):
        # Find the machine that allows this operation to finish the earliest.
        # Since all ready ops start at 'time', the finish time is time + duration.
        # So we just want the minimum duration for this specific operation.
        # However, if we want to be smarter about "machine contention" in a lookahead sense:
        # A simple 1-step lookahead assumes the machine is available. 
        # If we want to account for other ready ops potentially competing for the same machine,
        # that requires a 2-step lookahead or a more complex model.
        # Given the constraints and the "different strategy" directive, let's stick to a 
        # robust 1-step lookahead with strong tie-breakers, which is effectively a 
        # "Shortest Processing Time First" (SPTF) but applied per-operation with job-awareness.
        
        # Actually, let's refine the "Different Strategy": 
        # Current: Weighted sum of static features.
        # New: Dynamic Priority based on "Critical Path Slack" approximation.
        # Slack = (Remaining Work) - (Min Duration of Next Op).
        # High slack means we have time; low slack means we are on the critical path.
        # We should prioritize ops with LOW slack (critical) or HIGH remaining work (big impact).
        
        # Let's try a different angle: "Most Critical Job First" using a lookahead of the 
        # remaining operations' minimum durations.
        
        # For each op, calculate:
        # 1. min_duration: The shortest time this op could take (on any machine).
        # 2. projected_finish: time + min_duration.
        
        # But wait, the current program already does something similar with "availability".
        # Let's try a true **Lookahead Simulation** where we estimate the makespan impact.
        
        # Alternative Different Strategy: **Nearest Neighbor / Greedy Local Search**
        # Pick the op that minimizes the *maximum* of the current time and the machine's 
        # projected load, considering only the ready queue.
        
        # Let's go with a **Dynamic Critical Path** approach:
        # 1. For each ready op, find the machine that minimizes the finish time (min duration).
        # 2. Estimate the "criticality" of the job.
        #    Criticality = sum of min durations of all remaining ops in the job (including this one).
        #    This is a lower bound on the job's remaining time.
        # 3. Prioritize the op belonging to the job with the HIGHEST criticality (most work left).
        # 4. Tie-break by the finish time of the op itself (earliest finish).
        
        # This is different from the current "weighted sum" because it separates the 
        # job-level priority (criticality) from the operation-level timing (finish time).
        
        # Calculate job criticality (lower bound on remaining time for the job)
        # We don't have the full job data, only 'remaining_work' and 'remaining_ops'.
        # 'remaining_work' is likely the sum of durations. 
        # Let's use 'remaining_work' as the primary job priority signal.
        
        # New Strategy: **Two-Tier Priority**
        # Tier 1: Job Urgency (remaining_work)
        # Tier 2: Operation Efficiency (min duration for this op)
        
        # But the current program does alpha * rw + ... which is also a weighted sum.
        
        # Let's try **Simulated Annealing-like randomization with greedy acceptance**?
        # No, deterministic is safer for scoring stability.
        
        # Let's try **Min-Max Regret**:
        # For each op, calculate the "regret" of NOT picking it.
        # Regret = (Time if picked) - (Time if not picked).
        # This is hard to define simply.
        
        # Let's go back to the **Lookahead** idea but make it concrete:
        # We will simulate the next event.
        # If we pick op i, it finishes at time + min_duration(i).
        # We want to pick the op that leads to the earliest *potential* completion of *any* job.
        
        # Actually, the most materially different strategy from a "weighted heuristic" is 
        # a **Greedy Lookahead** that evaluates the *state* after dispatch.
        
        # Since we can't easily simulate the full state, let's use a **Proxy for Machine Contention**
        # that is different from the current "sum of loads".
        
        # Current: machine_load[m] = sum of durations of ready ops on m.
        # New: machine_contention[m] = number of ready ops that CAN run on m (flexibility penalty).
        # Ops that have only one machine option are "stiff". Ops with many options are "flexible".
        # We should prioritize "stiff" ops because if we don't run them now, they might block a machine 
        # that no other op can use, or they have no alternative.
        # This is a **Flexibility-Based Dispatching** rule.
        
        # Rule:
        # 1. Calculate flexibility for each op: len(machine_options).
        # 2. Prioritize ops with LOWEST flexibility (fewest options).
        # 3. Tie-break by remaining_work (urgency).
        # 4. Tie-break by min_duration (efficiency).
        
        # This is fundamentally different from the current load-balancing weighted sum.
        # It addresses the "bottleneck" risk of rigid jobs.
        
        min_flex = float('inf')
        best_idx = 0
        best_rw = -1
        best_min_dur = float('inf')
        
        for i, op in enumerate(ready_operations):
            flex = len(op["machine_options"])
            rw = op["remaining_work"]
            min_dur = min(op["machine_options"].values())
            
            # We want to minimize flex, then maximize rw, then minimize min_dur
            # Let's create a tuple for comparison
            # (flex, -rw, min_dur)
            # We want the lexicographically smallest tuple.
            
            current_key = (flex, -rw, min_dur)
            
            if best_key is None or current_key < best_key:
                best_key = current_key
                best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
