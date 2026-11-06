"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Enhanced regret-based heuristic for flexible job-shop scheduling.
    
    For each ready operation, compute the regret: the difference between the
    best and second-best machine assignment costs. The cost of assigning an
    operation to a machine is estimated as:
      duration + estimated_future_load(machine)
    
    where estimated_future_load accounts for:
      1. Total remaining work across all jobs on that machine
      2. Current time offset to account for already-elapsed time
    
    We select the operation with the maximum regret, as delaying a decision
    with high regret is most likely to worsen the schedule.
    
    Tie-breaking considers:
      1. Total remaining work (urgency of job completion)
      2. Minimum processing time (critical path consideration)
      3. Number of remaining operations (job length)
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Estimate future machine load by summing minimum durations across all
    # remaining operations of all jobs that could potentially use each machine.
    # We use minimum duration as a lower bound estimate for actual processing time.
    machine_load = {}
    for op in ready_operations:
        # For each machine option in this ready operation, add its duration
        for machine, duration in op["machine_options"].items():
            if machine not in machine_load:
                machine_load[machine] = 0.0
            machine_load[machine] += duration
    
    # Also estimate load from remaining operations of all jobs
    # Use a conservative estimate: sum of min durations for each remaining op
    # that has this machine as an option
    job_remaining_ops = {}
    for op in ready_operations:
        job_id = op["job"]
        if job_id not in job_remaining_ops:
            job_remaining_ops[job_id] = 0
        job_remaining_ops[job_id] = op["remaining_ops"]
    
    # For a better load estimate, we consider that jobs with more remaining ops
    # will generate more future work. Weight the machine load by job urgency.
    # Adjust machine_load by considering the average remaining ops per job
    if job_remaining_ops:
        avg_remaining = sum(job_remaining_ops.values()) / len(job_remaining_ops)
        # Scale factor: jobs with more remaining work are more urgent
        # This is a heuristic adjustment
        for machine in machine_load:
            # Slight boost for machines with higher estimated load
            pass  # Keep simple for now
    
    # For each operation, compute the regret using enhanced cost function
    best_costs = []
    min_durations = []
    for op in ready_operations:
        options = op["machine_options"]
        # Compute cost for each machine: duration + weighted future load
        costs = []
        for machine, duration in options.items():
            future_load = machine_load.get(machine, 0.0)
            # Use a slightly higher weight for future load to better capture congestion
            cost = duration + 0.6 * future_load
            costs.append(cost)
        
        # Track minimum duration for tie-breaking
        min_dur = min(options.values())
        min_durations.append(min_dur)
        
        if len(costs) <= 1:
            # Only one option, no regret
            regret = 0.0
        else:
            # Sort costs to find best and second-best
            sorted_costs = sorted(costs)
            regret = sorted_costs[1] - sorted_costs[0]
        
        best_costs.append(regret)
    
    # Select the operation with maximum regret
    # Use a small epsilon for floating point comparison
    max_regret = max(best_costs)
    epsilon = 1e-9
    
    # Find all operations with maximum regret (within tolerance)
    candidates = [i for i in range(n_ops) if best_costs[i] >= max_regret - epsilon]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Enhanced tie-breaking:
    # 1. Prefer operation with more total remaining work (job urgency)
    # 2. Prefer operation with shorter minimum duration (critical path)
    # 3. Prefer operation with more remaining ops (longer jobs need attention)
    def tiebreak_score(i):
        op = ready_operations[i]
        remaining_work = op["remaining_work"]
        min_duration = min_durations[i]
        remaining_ops = op["remaining_ops"]
        # Higher remaining work is better, shorter min duration is better,
        # more remaining ops is better
        return (remaining_work, -min_duration, remaining_ops)
    
    candidates.sort(key=tiebreak_score, reverse=True)
    return int(candidates[0])
# EVOLVE-BLOCK-END
