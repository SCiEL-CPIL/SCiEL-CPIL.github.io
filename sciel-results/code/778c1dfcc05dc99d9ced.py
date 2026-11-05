"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_start_time = _time.time()
_DEADLINE = 145.0  # 5s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: Greedy Look-Ahead (Gla).
    
    Strategy: For each candidate operation, simulate its completion and estimate the 
    earliest possible completion time of the next operation(s) that would become ready.
    This shifts the focus from local processing time (SPT/NSPT) to global bottleneck 
    progression, which is often more critical in job-shop scheduling to avoid 
    machine contention and idle time.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    best_score = float('inf')
    best_idx = 0

    # Pre-compute min durations for all ready ops to speed up look-ahead
    min_durations = []
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_durations.append(min(machine_options.values()))
        else:
            min_durations.append(float('inf'))

    for i, op in enumerate(ready_operations):
        # Estimate completion time of this operation
        # Using min duration as a lower bound proxy for actual duration
        comp_time = time + min_durations[i]
        
        # Look ahead: What is the earliest completion time among the next ready ops?
        # We approximate the "next ready ops" by checking:
        # 1. Other currently ready ops (they can start at 'time', so their completion is time + their duration)
        # 2. The next op of the current job (starts at comp_time, so completion is comp_time + next_duration)
        
        # Note: In a real job shop, other ready ops might be delayed by machine contention.
        # However, for a 1-step look-ahead heuristic, assuming they can proceed concurrently 
        # (or at least that their potential completion is an indicator of system load) is standard.
        
        # The "bottleneck" of the next state is roughly the min completion time among all potential next ops.
        # If we pick op i, the next potential completions are:
        # - All other ready ops j != i: complete at time + min_durations[j]
        # - The next op of job i: completes at comp_time + next_duration_of_job_i
        
        # We want to minimize the time until the system makes progress.
        # A common proxy is the minimum completion time among these.
        
        min_next_completion = float('inf')
        
        # Check other currently ready ops
        for j, other_op in enumerate(ready_operations):
            if j == i:
                continue
            # These ops are ready now. Their completion is time + duration.
            # They will be ready to complete at this time regardless of which op we pick now 
            # (assuming machines are available, which is an optimistic assumption for the heuristic).
            potential_completion = time + min_durations[j]
            if potential_completion < min_next_completion:
                min_next_completion = potential_completion
        
        # Check the next op of the current job
        remaining_ops = op.get("remaining_ops", [])
        if remaining_ops:
            # The next op is the first in the remaining list
            next_op = remaining_ops[0]
            next_machine_options = next_op.get("machine_options", {})
            if next_machine_options:
                next_duration = min(next_machine_options.values())
                # This op can start at comp_time
                potential_completion = comp_time + next_duration
                if potential_completion < min_next_completion:
                    min_next_completion = potential_completion
        
        # If no other ops and no next op, the completion time itself is the score
        if min_next_completion == float('inf'):
            min_next_completion = comp_time

        # We want to minimize this score (earliest progress)
        if min_next_completion < best_score:
            best_score = min_next_completion
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
