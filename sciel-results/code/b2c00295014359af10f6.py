"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

def dispatch(ready_operations, time):
    """
    Dispatch rule using Dynamic Machine Load Balancing.
    
    Strategy:
    1. Estimate the immediate future load on each machine by summing the minimum
       processing times of all currently ready operations that can run on that machine.
    2. For each ready operation, calculate a "bottleneck score" based on the load
       of the machines it can use. We want to schedule operations on machines that
       are currently less crowded to avoid creating a bottleneck.
    3. Among operations that can run on the least loaded machines, we prioritize
       the job with the most remaining work (MWR) to ensure critical jobs progress.
    
    This differs from static MWR by actively trying to balance machine utilization
    in the immediate step, which can reduce makespan by preventing one machine from
    becoming a severe bottleneck while others are idle.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    timeout = 0.1  # 100ms limit for this dispatch call to be safe
    
    # Step 1: Calculate the estimated load on each machine from the current pool of ready ops.
    # We assume that if a machine is in 'machine_options', the op *can* run there.
    # The "load" is the sum of the minimum durations for all ops that can run on that machine.
    # This is a proxy for how busy that machine will be if we don't manage it well.
    
    machine_loads = {}
    for op in ready_operations:
        options = op["machine_options"]
        for machine, duration in options.items():
            if machine not in machine_loads:
                machine_loads[machine] = 0.0
            # Use the specific duration for this machine, not min, because if we assign it here,
            # it takes 'duration'. But for "congestion" estimation, we just want to know
            # how much work is *available* for this machine.
            # Actually, a better proxy for "which machine is free" is to look at the 
            # minimum duration across all ops that *must* use it? No, we don't have that info.
            
            # Let's use the duration of this specific op on this machine as its contribution
            # to the machine's potential load.
            machine_loads[machine] += duration

    # Step 2: Score each operation.
    # We want to pick an op that runs on a machine with LOW load (to balance).
    # But we also want to respect the job's remaining work (MWR).
    
    # Let's define a composite score:
    # Primary: Minimize the load of the machine we are assigning to.
    # Secondary: Maximize remaining_work.
    
    # Since we return an index, we need a single scalar score per op.
    # Let's try: Score = remaining_work * (1 / (min_machine_load_for_this_op + epsilon))
    # This heavily weights jobs with high remaining work, but boosts them if they can run on a less loaded machine.
    
    epsilon = 1e-9
    min_load = min(machine_loads.values()) if machine_loads else 1.0
    max_load = max(machine_loads.values()) if machine_loads else 1.0
    
    # Normalize loads to [0, 1] where 0 is least loaded, 1 is most loaded.
    # We want to favor LOW load. So (1 - normalized_load) is the "freedom" score.
    
    scores = []
    for i, op in enumerate(ready_operations):
        if _time.time() - start_time > timeout:
            # Fallback to simple MWR if we are taking too long
            return int(np.argmax([o["remaining_work"] for o in ready_operations]))
            
        options = op["machine_options"]
        rem_work = op["remaining_work"]
        
        # Find the machine with the lowest load among those this op can use.
        # This represents the "best" machine for this op in terms of balance.
        min_load_for_op = min(machine_loads.get(m, 0.0) for m in options.keys())
        
        # Normalize this load relative to the global min/max to get a "freedom" factor.
        # If all loads are the same, freedom is 1.0.
        if max_load > min_load:
            # Map min_load -> 1.0 (best), max_load -> 0.0 (worst)
            freedom = 1.0 - (min_load_for_op - min_load) / (max_load - min_load)
        else:
            freedom = 1.0
            
        # Combine: We want high remaining work and high freedom.
        # Let's try a multiplicative combination: rem_work * freedom
        # Or additive: rem_work + alpha * freedom
        # Multiplicative might be too harsh if freedom is 0.
        # Let's use: rem_work * (0.5 + 0.5 * freedom)
        # This ensures rem_work is the primary driver, but freedom can boost it by up to 50%
        # or reduce it by 50%.
        
        score = rem_work * (0.5 + 0.5 * freedom)
        scores.append(score)

    return int(np.argmax(scores))

# EVOLVE-BLOCK-END
