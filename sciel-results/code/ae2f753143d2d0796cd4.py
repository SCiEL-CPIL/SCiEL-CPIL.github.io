"""
Strategy: Bottleneck-Aware Dispatch with Machine Load Balancing
Instead of just picking the job with most remaining work, we compute a composite score
that balances job urgency (remaining work) against machine congestion.
We track machine load dynamically and prefer machines that are less busy.
"""

import time as _time
import numpy as np

# Global state
_machine_load = {}  # machine -> current load (total time assigned so far)
_machine_free_at = {}  # machine -> time when it becomes free
_instance_id_counter = 0
_last_time_seen = -1.0

def dispatch(ready_operations, time):
    global _machine_load, _machine_free_at, _instance_id_counter, _last_time_seen
    
    # Detect new instance: if time drops significantly or is 0, reset state
    # In event-driven simulation, time is non-decreasing within an instance.
    # If we see time < _last_time_seen - 1e-9, it's a new instance.
    if time < _last_time_seen - 1e-9:
        _machine_load = {}
        _machine_free_at = {}
        _instance_id_counter += 1
    _last_time_seen = time

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Compute scores for each ready operation
    # Score = (remaining_work) / (1 + machine_congestion_penalty)
    # machine_congestion_penalty is based on the load of the best available machine for this op
    
    best_idx = -1
    best_score = -1e18
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        machine_options = op["machine_options"]
        
        if not machine_options:
            # No machine options? Skip or handle edge case
            continue
            
        # For this operation, find the machine that minimizes:
        # max(time, machine_free_at[machine]) + duration
        # But we also want to balance load.
        # Let's compute a "cost" for each machine option.
        
        min_cost = float('inf')
        chosen_machine = None
        
        for machine, duration in machine_options.items():
            # Estimate when this machine will be free
            free_at = _machine_free_at.get(machine, 0.0)
            # The operation can start at max(time, free_at)
            # But if we assign it now, it will finish at max(time, free_at) + duration
            # However, the simulator will start it at 'time' if the machine is free,
            # or wait? The interface says "picks which ready operation to start next".
            # Typically, if you pick an op, it starts on the chosen machine at 'time'
            # only if the machine is free. If not, it might be queued?
            # The prompt says "start next", implying it starts immediately if possible.
            # If the machine is busy, does the op wait? Or does the simulator pick another?
            # Usually in these dispatch rules, you pick an op and a machine.
            # If the machine is busy, the op is queued on that machine?
            # Or does the rule only pick ops that can start *now*?
            # "picks which ready operation to start next in an event-driven... simulation"
            # Event-driven usually means when an event happens (machine free), you dispatch.
            # So `time` is the current event time.
            # If I pick an op for a machine that is free at `time`, it starts.
            # If I pick an op for a machine that is busy until `t > time`, 
            # does it start at `t`? Or is that invalid?
            # The return interface is `(index, machine)`.
            # Let's assume the simulator handles the queuing or we should pick
            # the machine that is free earliest to minimize delay.
            
            # Heuristic: Prefer the machine that minimizes the completion time
            # of this specific operation, but also consider the overall load.
            
            start_time = max(time, free_at)
            completion_time = start_time + duration
            
            # Add a penalty if this machine is already heavily loaded
            # Load balance term: we want to avoid overloading one machine
            load_penalty = _machine_load.get(machine, 0.0)
            
            # Composite cost: completion_time + alpha * load_penalty
            # We want to minimize this cost.
            # But we also want to prioritize jobs with high remaining work.
            # So, Score = remaining_work / (1 + normalized_cost)
            
            cost = completion_time + 0.1 * load_penalty
            
            if cost < min_cost:
                min_cost = cost
                chosen_machine = machine
        
        if chosen_machine is None:
            continue
            
        # The score for this operation is its remaining work,
        # but discounted by how "expensive" it is to schedule on its best machine.
        # If the cost is high (machine busy), the score drops.
        
        # Normalize cost? Costs can vary widely.
        # Let's use a simpler heuristic:
        # Score = remaining_work - beta * min_cost
        
        beta = 0.5
        score = remaining_work - beta * min_cost
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine

    if best_idx == -1:
        # Fallback: pick the first one
        best_idx = 0
        best_machine = list(ready_operations[0]["machine_options"].keys())[0]

    # Update state
    machine = best_machine
    duration = ready_operations[best_idx]["machine_options"][machine]
    
    # The operation will start at max(time, _machine_free_at.get(machine, 0))
    # But for load tracking, we add the duration to the load.
    # And update the free time.
    
    current_free = _machine_free_at.get(machine, 0.0)
    start_time = max(time, current_free)
    finish_time = start_time + duration
    
    _machine_free_at[machine] = finish_time
    _machine_load[machine] = _machine_load.get(machine, 0.0) + duration

    return (best_idx, machine)
