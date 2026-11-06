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
       of the machines it can use. We want to schedule operations that have access 
       to less congested machines, or that are critical enough to warrant the wait.
    3. This is a more dynamic approach than static MWR/SPT because it accounts 
       for the interactions between ready operations competing for the same resources 
       in the next time step.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Deadline guard
    start_time = _time.time()
    deadline = 110.0  # Safety margin for 120s limit

    # Extract machine options and durations
    # Structure: ready_operations[i] has 'machine_options' {machine: duration}
    
    # Step 1: Calculate "potential load" for each machine.
    # We assume that if multiple jobs are ready, the machine will be busy with 
    # the sum of durations of jobs assigned to it. 
    # Since we don't know the assignment yet, we estimate the "pressure" on each machine.
    
    # A simpler, robust proxy for "machine congestion" in the next step:
    # For each machine, sum the minimum possible duration of all ready ops that can use it.
    # This represents the total work that *could* be dumped on that machine.
    
    machine_load = {}
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            if m not in machine_load:
                machine_load[m] = 0
            machine_load[m] += dur

    # Step 2: Score each ready operation.
    # We want to prioritize operations that:
    # a) Have high remaining work (MWR-like)
    # b) Can be scheduled on a machine with relatively low "potential load" 
    #    OR have a short duration on their best machine, reducing the impact of congestion.
    
    # Let's define the score as: 
    # Priority = remaining_work / (1 + min_machine_load_for_this_op)
    # This penalizes operations that are forced to use heavily loaded machines, 
    # unless their remaining work is very high.
    
    # Alternatively, a "waiting time" simulation:
    # For each op, find the machine with the minimum (current_time + estimated_wait + duration).
    # But we don't have current_time per machine, only global 'time'.
    # In a list-scheduling algorithm, the machine is available at 'time' if it's idle.
    # The "wait" is caused by other ops.
    
    # Let's try a different angle: **Least Congested Machine Priority**.
    # For each ready operation, identify its best machine (min duration).
    # Check how "busy" that machine is (based on the sum of durations of other ready ops on it).
    # Score = remaining_work - penalty * (load_on_best_machine / max_load)
    
    scores = []
    min_duration_per_op = []
    best_machine_per_op = []
    
    for op in ready_operations:
        opts = op["machine_options"]
        if not opts:
            scores.append(0)
            min_duration_per_op.append(float('inf'))
            best_machine_per_op.append(None)
            continue
        
        # Find the machine with the shortest duration for this op
        best_m = min(opts, key=opts.get)
        min_dur = opts[best_m]
        
        min_duration_per_op.append(min_dur)
        best_machine_per_op.append(best_m)
        
        # Calculate load on this best machine from OTHER ready operations
        # This is an estimate of contention.
        load_on_best_m = 0
        for other_op in ready_operations:
            if other_op is op:
                continue
            if best_m in other_op["machine_options"]:
                # Add the duration of the other op on this machine
                # Using min duration as a proxy for "how much time it takes"
                load_on_best_m += other_op["machine_options"][best_m]
        
        rem_work = op["remaining_work"]
        
        # If rem_work is not available or 0, use min_dur as a fallback proxy
        if rem_work is None or rem_work == 0:
             rem_work = min_dur
        
        # Heuristic: 
        # High remaining work is good.
        # High load on the best machine is bad (contention).
        # Short min duration is good (quick release).
        
        # Score = rem_work - alpha * load_on_best_m
        # We need to normalize load_on_best_m to be comparable to rem_work.
        # Max possible load is sum of all durations on that machine.
        
        # Let's try a ratio:
        # Score = rem_work / (1.0 + load_on_best_m / (min_dur + 1.0))
        # This penalizes high contention relative to the op's own speed.
        
        # Simpler: 
        # Score = rem_work * (1.0 / (1.0 + load_on_best_m * 0.1))
        
        # Let's use a "urgency" metric:
        # If a machine is heavily loaded, we should avoid it unless the job is critical.
        # Criticality = rem_work.
        
        # Score = rem_work - 0.5 * load_on_best_m
        # This is linear and simple. Let's try to balance it.
        
        # Actually, let's look at the "effective completion time" if we start this op now.
        # If we start it, it finishes at time + min_dur.
        # The "cost" is blocking the machine for min_dur.
        # The "benefit" is reducing remaining_work by min_dur (or more if rem_work includes it).
        
        # Let's stick to the contention penalty.
        score = rem_work - 0.5 * load_on_best_m
        
        scores.append(score)

    # Check time
    if _time.time() > deadline:
        # Fallback to simple MWR
        rem_works = [op["remaining_work"] if op["remaining_work"] else 0 for op in ready_operations]
        return int(np.argmax(rem_works))

    return int(np.argmax(scores))

# EVOLVE-BLOCK-END
