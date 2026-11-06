import time
import math
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _get_machine_and_duration(op):
    """Get the machine and duration for an operation.
    In open shop, each operation has exactly one machine option.
    """
    for m, d in op["machine_options"].items():
        return m, d
    return None, 0

def _fast_greedy_sim(ready_operations, time, chosen_index, instance_ops, completed_ops, machine_free_time):
    """Fast greedy simulation to estimate makespan after choosing a specific operation.
    Uses a simple priority: earliest machine availability, tie-break by shortest duration.
    """
    # Copy state
    sim_mft = dict(machine_free_time)
    sim_completed = set(completed_ops)
    sim_ops = dict(instance_ops)
    
    # Update instance ops with current ready ops
    for op in ready_operations:
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        if (job, m) not in sim_ops:
            sim_ops[(job, m)] = d
    
    # Start the chosen operation
    chosen_op = ready_operations[chosen_index]
    m, d = _get_machine_and_duration(chosen_op)
    job = chosen_op["job"]
    
    sim_mft[m] = time + d
    sim_completed.add((job, m))
    
    # Get pending ops
    pending = []
    for (j, mm), dd in sim_ops.items():
        if (j, mm) not in sim_completed:
            pending.append((j, mm, dd))
    
    if not pending:
        return sim_mft[m]
    
    max_end = sim_mft[m]
    
    # Greedy: pick op with earliest machine availability, tie-break SPT
    while pending:
        best_idx = 0
        best_start = sim_mft[pending[0][1]]
        best_dur = pending[0][2]
        
        for i in range(1, len(pending)):
            j, mm, dd = pending[i]
            start_t = sim_mft[mm]
            if start_t < best_start or (start_t == best_start and dd < best_dur):
                best_start = start_t
                best_dur = dd
                best_idx = i
        
        j, mm, dd = pending[best_idx]
        end_t = best_start + dd
        sim_mft[mm] = end_t
        sim_completed.add((j, mm))
        pending.pop(best_idx)
        
        if end_t > max_end:
            max_end = end_t
    
    return max_end

def _multi_criteria_score(ready_operations, time, instance_ops, completed_ops, machine_free_time):
    """Score each ready operation using multiple criteria.
    Returns a list of scores (lower is better).
    """
    scores = []
    n = len(ready_operations)
    
    for i in range(n):
        op = ready_operations[i]
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        
        # Criterion 1: Machine load (remaining work on this machine)
        machine_load = 0.0
        for (j, mm), dd in instance_ops.items():
            if mm == m and (j, mm) not in completed_ops:
                machine_load += dd
        
        # Criterion 2: Job remaining work (MWR - Most Work Remaining)
        job_remaining = op.get("remaining_work", d)
        
        # Criterion 3: Duration (SPT - Shortest Processing Time)
        duration = d
        
        # Criterion 4: Machine free time (earliest available)
        mft = machine_free_time.get(m, 0.0)
        
        # Combine: prefer operations that reduce critical path
        # Weighted sum with emphasis on machine load and job remaining work
        score = 0.3 * (machine_load / (d + 1e-9)) + 0.3 * (job_remaining / (d + 1e-9)) + 0.2 * (d / (max(duration) + 1e-9) if max(duration) > 0 else 1.0) + 0.2 * (mft / (time + 1e-9) if time > 0 else 1.0)
        
        scores.append(score)
    
    return scores

def _local_search(ready_operations, time, initial_index, instance_ops, completed_ops, machine_free_time, num_iterations=50):
    """Perform local search around the initial choice to find a better makespan.
    Uses simulated annealing-style exploration.
    """
    n = len(ready_operations)
    if n <= 1:
        return initial_index
    
    best_index = initial_index
    best_makespan = _fast_greedy_sim(ready_operations, time, initial_index, instance_ops, completed_ops, machine_free_time)
    
    current_index = initial_index
    current_makespan = best_makespan
    
    temperature = 1.0
    cooling_rate = 0.95
    
    for it in range(num_iterations):
        if _time_left() < 5.0:
            break
        
        # Pick a random neighbor
        neighbor = random.randint(0, n - 1)
        if neighbor == current_index:
            continue
        
        neighbor_makespan = _fast_greedy_sim(ready_operations, time, neighbor, instance_ops, completed_ops, machine_free_time)
        
        delta = neighbor_makespan - current_makespan
        
        # Accept if better, or with probability based on temperature
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_index = neighbor
            current_makespan = neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_index = current_index
        
        temperature *= cooling_rate
    
    return best_index

def dispatch(ready_operations, time):
    """Main dispatch function that picks which ready operation to start next.
    
    Args:
        ready_operations: List of dicts, each with job, op, machine_options, remaining_ops, remaining_work
        time: Current time in the simulation
    
    Returns:
        Index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    # Build instance knowledge from ready operations
    instance_ops = {}
    completed_ops = set()
    machine_free_time = {}
    
    # We need to track state across calls. Use a simple approach:
    # Infer from ready_operations what's been completed.
    # An operation is completed if it's not in any ready_operations list we've seen.
    # For simplicity, we'll maintain global state.
    
    global _global_instance_ops
    global _global_completed_ops
    global _global_machine_free_time
    global _global_initialized
    
    if not _global_initialized:
        _global_instance_ops = {}
        _global_completed_ops = set()
        _global_machine_free_time = {m: 0.0 for m in range(5)}
        _global_initialized = True
    
    # Update instance ops
    for op in ready_operations:
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        if (job, m) not in _global_instance_ops:
            _global_instance_ops[(job, m)] = d
    
    instance_ops = _global_instance_ops
    completed_ops = _global_completed_ops
    machine_free_time = _global_machine_free_time
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        op = ready_operations[0]
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        machine_free_time[m] = time + d
        completed_ops.add((job, m))
        return 0
    
    # Time budget check
    if _time_left() < 10.0:
        # Fallback to simple heuristic
        best_idx = 0
        best_score = float('inf')
        for i in range(n):
            op = ready_operations[i]
            m, d = _get_machine_and_duration(op)
            score = d + machine_free_time.get(m, 0.0)
            if score < best_score:
                best_score = score
                best_idx = i
        op = ready_operations[best_idx]
        m, d = _get_machine_and_duration(op)
        job = op["job"]
        machine_free_time[m] = time + d
        completed_ops.add((job, m))
        return best_idx
    
    # Step 1: Get initial candidate using multi-criteria scoring
    scores = _multi_criteria_score(ready_operations, time, instance_ops, completed_ops, machine_free_time)
    initial_index = scores.index(min(scores))
    
    # Step 2: Local search to improve
    if _time_left() > 15.0:
        best_index = _local_search(ready_operations, time, initial_index, instance_ops, completed_ops, machine_free_time, num_iterations=30)
    else:
        best_index = initial_index
    
    # Update global state with the choice
    chosen_op = ready_operations[best_index]
    m, d = _get_machine_and_duration(chosen_op)
    job = chosen_op["job"]
    
    machine_free_time[m] = time + d
    completed_ops.add((job, m))
    
    return best_index

# Global state initialization
_global_instance_ops = {}
_global_completed_ops = set()
_global_machine_free_time = {}
_global_initialized = False
