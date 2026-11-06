import numpy as np

# Global state to track the instance and schedule progress
_instance_ops = {}  # {(job, machine): duration}
_completed_ops = set()  # {(job, machine)}
_machine_free_time = {}  # {machine: time}
_initialized = False
_num_machines = 5
_num_jobs = 10

def _init_state():
    global _machine_free_time, _initialized
    if not _initialized:
        for m in range(_num_machines):
            _machine_free_time[m] = 0.0
        _initialized = True

def _get_machine_and_duration(op):
    # machine_options is a dict {machine: duration}
    # In open shop, it should have exactly one entry
    # But just in case, pick the first or the one with min duration?
    # Usually fixed.
    for m, d in op["machine_options"].items():
        return m, d
    return None, 0

def _simulate_makespan(ready_operations, time, chosen_index):
    # Copy state
    sim_machine_free_time = dict(_machine_free_time)
    sim_completed_ops = set(_completed_ops)
    sim_instance_ops = dict(_instance_ops)
    
    # Update instance ops with current ready ops (in case we missed some)
    for op in ready_operations:
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        if (job, m) not in sim_instance_ops:
            sim_instance_ops[(job, m)] = d

    # Start the chosen operation
    chosen_op = ready_operations[chosen_index]
    m, d = _get_machine_and_duration(chosen_op)
    job = chosen_op["job"]
    
    # The machine must be free at 'time'
    # sim_machine_free_time[m] should be <= time
    sim_machine_free_time[m] = time + d
    sim_completed_ops.add((job, m))
    
    # Get pending ops
    pending_ops = []
    for (j, mm), dd in sim_instance_ops.items():
        if (j, mm) not in sim_completed_ops:
            pending_ops.append((j, mm, dd))
            
    max_end_time = sim_machine_free_time[m]
    
    # Greedy simulation: SPT among available
    while pending_ops:
        # Find the op with the earliest machine availability
        # If tie, pick shortest duration (SPT)
        # We need to check machine_free_time for each op
        best_op = None
        best_start = float('inf')
        best_dur = float('inf')
        
        for (j, mm, dd) in pending_ops:
            start_t = sim_machine_free_time[mm]
            if start_t < best_start or (start_t == best_start and dd < best_dur):
                best_start = start_t
                best_dur = dd
                best_op = (j, mm, dd)
                
        if best_op is None:
            break
            
        j, mm, dd = best_op
        end_t = best_start + dd
        sim_machine_free_time[mm] = end_t
        sim_completed_ops.add((j, mm))
        pending_ops.remove(best_op)
        
        if end_t > max_end_time:
            max_end_time = end_t
            
    return max_end_time

def dispatch(ready_operations, time):
    global _instance_ops, _completed_ops, _machine_free_time
    
    _init_state()
    
    # Update instance knowledge
    for op in ready_operations:
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        if (job, m) not in _instance_ops:
            _instance_ops[(job, m)] = d
            
    # Identify completed ops based on time
    # If an op was started and time >= free_time, it's done?
    # We don't track "started" explicitly, but we add to _completed_ops when we dispatch.
    # So _completed_ops is accurate for dispatched ops.
    
    if not ready_operations:
        return 0

    # Evaluate each ready operation
    best_index = 0
    best_makespan = float('inf')
    
    for i in range(len(ready_operations)):
        makespan = _simulate_makespan(ready_operations, time, i)
        if makespan < best_makespan:
            best_makespan = makespan
            best_index = i
            
    # Update global state with the choice
    chosen_op = ready_operations[best_index]
    m, d = _get_machine_and_duration(chosen_op)
    job = chosen_op["job"]
    
    _machine_free_time[m] = time + d
    _completed_ops.add((job, m))
    
    return best_index
