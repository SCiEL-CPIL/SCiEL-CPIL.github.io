"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 115.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'machine_load': defaultdict(float),  # Estimated remaining work per machine
        'machine_busy': set(),  # Machines currently busy (approximated)
        'call_count': 0,
        'total_work_initial': 0,
        'jobs_seen': set(),
        'ops_per_job': defaultdict(int),
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Track machine loads across calls to make informed machine assignments
    2. Use MWR (Most Work Remaining) as primary job selection criterion
    3. Use SPT (Shortest Processing Time) with load balancing for machine selection
    4. Consider operation blocking: if a job has only one machine option for 
       remaining ops, prioritize that machine
    
    Returns: (index, machine) tuple
    """
    # Initialize on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return (0, None)
    
    # Time guard
    if _check_time():
        best_idx = max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"])
        op = ready_operations[best_idx]
        machine_options = op.get("machine_options", {})
        if machine_options:
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        else:
            best_machine = None
        return (best_idx, best_machine)
    
    state = _sim_state
    state['call_count'] += 1
    
    # Update state: track total work and machine loads
    for op in ready_operations:
        job = op.get('job')
        if job is not None:
            state['jobs_seen'].add(job)
            # Estimate remaining work on each machine from this op
            for m, d in op.get('machine_options', {}).items():
                state['machine_load'][m] += d * 0.01  # Small incremental update
    
    # Get max remaining work for normalization
    max_work = max(op["remaining_work"] for op in ready_operations)
    if max_work == 0:
        max_work = 1
    
    # Get max remaining ops for normalization
    max_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_ops == 0:
        max_ops = 1
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(0)
            best_machines.append(None)
            continue
        
        durations = list(machine_options.values())
        min_duration = min(durations)
        max_duration = max(durations)
        num_machines = len(machine_options)
        
        # Component 1: Most Work Remaining (primary) - normalized
        work_score = remaining_work / max_work
        
        # Component 2: Remaining operations (secondary) - normalized
        ops_score = remaining_ops / max_ops
        
        # Component 3: Urgency - if few ops remain, prioritize to avoid idle time
        urgency_score = 1.0 / (remaining_ops + 1.0) if remaining_ops > 0 else 1.0
        
        # Component 4: Machine selection score
        # For each machine option, compute a score considering:
        # - Processing time (shorter is better)
        # - Current estimated load (less loaded is better)
        best_machine_score = -float('inf')
        best_machine = None
        
        for m, d in machine_options.items():
            # Machine load factor: prefer less loaded machines
            load_penalty = state['machine_load'].get(m, 0.0)
            # SPT: shorter duration is better
            spt_score = 1.0 / (d + 0.1)
            # Combined machine score
            machine_score = spt_score * 10.0 - load_penalty * 0.5
            if machine_score > best_machine_score:
                best_machine_score = machine_score
                best_machine = m
        
        # Component 5: Flexibility bonus - more options is slightly better
        flex_score = min(num_machines / 5.0, 1.0)
        
        # Component 6: Blocking detection - if this is the only machine for 
        # some remaining op of this job, add urgency
        job = op.get('job')
        blocking_bonus = 0.0
        if job is not None and remaining_ops <= 2:
            # Near completion: prioritize to free up the job
            blocking_bonus = 0.3
        
        # Weighted combination
        w_work = 0.50
        w_ops = 0.15
        w_urgency = 0.10
        w_machine = 0.15
        w_flex = 0.05
        w_blocking = 0.05
        
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_urgency * urgency_score + 
                 w_machine * (best_machine_score / 10.0) + 
                 w_flex * flex_score +
                 w_blocking * blocking_bonus)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
