"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Strategy: Load-Aware Bottleneck Elimination
# 1. Track the estimated completion time (load) for each machine globally.
# 2. For each ready operation, calculate the job's criticality (remaining work).
# 3. For each eligible machine, estimate the operation's completion time 
#    (machine_load + duration).
# 4. Select the operation-machine pair that minimizes a composite cost:
#    - Primary: Minimize the job's remaining critical path (Urgency).
#    - Secondary: Minimize the estimated completion time on the chosen machine (Load balancing).
#    This avoids the "greedy fastest machine" trap where one machine gets
#    overloaded with short jobs while others sit idle, which is common in
#    flexible job shops.

_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit

# Global state to track machine loads
# We assume machine IDs are small integers or hashable. 
# We store the estimated time when the machine becomes free.
_machine_loads = {}

def _get_machine_load(machine_id):
    return _machine_loads.get(machine_id, 0)

def _update_machine_load(machine_id, completion_time):
    if completion_time > _machine_loads.get(machine_id, 0):
        _machine_loads[machine_id] = completion_time

def dispatch(ready_operations, time):
    # Check deadline
    if _time.time() - _START_TIME > _TIME_LIMIT:
        if not ready_operations:
            return 0
        # Fallback: pick the first one
        return 0

    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one operation is ready, we still need to pick a machine.
    # We pick the machine that minimizes the estimated completion time.
    if n == 1:
        op = ready_operations[0]
        best_machine = None
        best_completion = float('inf')
        for m, d in op["machine_options"].items():
            completion = _get_machine_load(m) + d
            if completion < best_completion:
                best_completion = completion
                best_machine = m
        # Update the load for the selected machine
        _update_machine_load(best_machine, best_completion)
        return (0, best_machine)
    
    best_idx = -1
    best_machine = None
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        job_id = op["job"]
        
        # Estimate job criticality
        # 'remaining_work' is assumed to be a proxy for total remaining processing time.
        # If 'remaining_ops' is a list, we could compute sum of mins, but 'remaining_work' 
        # is provided and likely accurate. We use it as the urgency.
        rem_work = op.get("remaining_work", 0)
        if not isinstance(rem_work, (int, float)):
            # Fallback if remaining_work is not a number
            rem_ops = op.get("remaining_ops", 0)
            if isinstance(rem_ops, list):
                rem_work = sum(min(r["machine_options"].values()) for r in rem_ops if isinstance(r, dict) and "machine_options" in r)
            else:
                rem_work = 0
        
        # For each eligible machine, calculate the cost
        # Cost = (Job Criticality * Weight_Urgency) + (Estimated Completion Time * Weight_Load)
        # We want to minimize this cost.
        # High criticality jobs should be scheduled early.
        # Low load machines should be preferred to balance the system.
        
        # We need to balance the scales. 
        # Criticality is in "time units" (sum of durations).
        # Completion time is also in "time units".
        # They are comparable. 
        # However, prioritizing the job with the most remaining work is crucial for makespan.
        # Balancing load prevents bottlenecks.
        
        # Let's try a score that prioritizes the job's criticality first, 
        # but among jobs with similar criticality, picks the one that finishes sooner on a machine.
        # Actually, a better metric for makespan reduction is often:
        # Score = Estimated Completion Time of this Op on Machine M
        # But we must respect job precedence urgency.
        
        # Alternative: 
        # 1. Find the machine that minimizes completion time for this specific op.
        # 2. Score the op based on: (Job Criticality) - alpha * (Completion Time on best machine)
        # Wait, minimizing completion time is good. Minimizing criticality is bad (we want to do critical jobs first).
        # So we want to MAXIMIZE Criticality and MINIMIZE Completion Time.
        
        # Let's define a composite score to MAXIMIZE:
        # Score = Criticality - lambda * (Completion Time on Best Machine)
        # This favors jobs with lots of work left, but breaks ties by picking the one that finishes sooner.
        
        best_machine_for_op = None
        best_completion_for_op = float('inf')
        
        for m, d in op["machine_options"].items():
            completion = _get_machine_load(m) + d
            if completion < best_completion_for_op:
                best_completion_for_op = completion
                best_machine_for_op = m
        
        # The "cost" of scheduling this op now is that it delays the job's completion.
        # But we want to schedule the "most urgent" job first.
        # So we prioritize high 'rem_work'.
        
        # Let's use a score where we prefer:
        # 1. High remaining work (Urgency)
        # 2. Low estimated completion time on the best available machine (Efficiency/Load Balancing)
        
        # Score = rem_work * 10 - best_completion_for_op
        # The multiplier 10 ensures that a job with significantly more work is always preferred,
        # unless the completion time difference is huge.
        
        score = rem_work * 10.0 - best_completion_for_op
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_machine_for_op
            
    if best_idx == -1:
        # Fallback
        best_idx = 0
        op = ready_operations[0]
        best_machine = min(op["machine_options"].keys(), key=lambda m: _get_machine_load(m) + op["machine_options"][m])
        
    # Update the global machine load for the selected machine
    selected_op = ready_operations[best_idx]
    selected_duration = selected_op["machine_options"][best_machine]
    completion_time = _get_machine_load(best_machine) + selected_duration
    _update_machine_load(best_machine, completion_time)
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
