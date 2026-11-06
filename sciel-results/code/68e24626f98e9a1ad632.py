"""Optimized dispatch rule for open-shop scheduling with load-balancing water-filling."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

# Global state for tracking simulation progress
_sim_state = {}
_DEADLINE = 110.0
_start_time = None

def _reset_sim_state():
    global _sim_state, _start_time
    _sim_state = {
        'machine_load': defaultdict(float),      # Estimated remaining work per machine
        'machine_total_work': defaultdict(float),# Total work seen per machine
        'machine_assigned': defaultdict(float),  # Work already assigned to machine
        'call_count': 0,
        'jobs_seen': set(),
        'ops_per_job': defaultdict(int),
        'last_time': 0.0,
        'total_machines_seen': set(),
    }
    _start_time = time.time()

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def _soft_min(values, temp=1.0):
    """Smooth minimum via log-sum-exp trick."""
    if not values:
        return 0.0
    v = min(values)
    if temp <= 0:
        return v
    exp_sum = sum(math.exp(-(x - v) / temp) for x in values)
    return v - temp * math.log(exp_sum)

def _soft_max(values, temp=1.0):
    """Smooth maximum via log-sum-exp trick."""
    if not values:
        return 0.0
    v = max(values)
    if temp <= 0:
        return v
    exp_sum = sum(math.exp((x - v) / temp) for x in values)
    return v + temp * math.log(exp_sum)

def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Load-Balancing Water-Filling
    - For each candidate operation, simulate assignment to the least-loaded compatible machine
    - Score by resulting increase in maximum machine load (L-infinity norm)
    - Prioritize operations that minimally skew the global workload distribution
    - Use MWR as tiebreaker for job selection
    
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
    
    # Update state: track machine loads based on elapsed time
    dt = max(0.0, time - state['last_time'])
    state['last_time'] = time
    
    for op in ready_operations:
        job = op.get('job')
        if job is not None:
            state['jobs_seen'].add(job)
        # Track total work seen on each machine
        for m, d in op.get('machine_options', {}).items():
            state['machine_total_work'][m] += d
            state['total_machines_seen'].add(m)
    
    # Get max remaining work for normalization
    max_work = max(op["remaining_work"] for op in ready_operations)
    if max_work == 0:
        max_work = 1
    
    # Get max remaining ops for normalization
    max_ops = max(op["remaining_ops"] for op in ready_operations)
    if max_ops == 0:
        max_ops = 1
    
    # Get current machine loads (estimated remaining work)
    # We approximate: total_work_seen - assigned_work = remaining
    # For simplicity, use total_work as proxy for load
    total_all_work = sum(state['machine_total_work'].values())
    if total_all_work == 0:
        total_all_work = 1
    
    # Current max load across all known machines
    all_machines = state['total_machines_seen']
    current_loads = {m: state['machine_total_work'].get(m, 0.0) for m in all_machines}
    current_max_load = max(current_loads.values()) if current_loads else 0.0
    if current_max_load == 0:
        current_max_load = 1
    
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
        
        # Component 4: Load-Balancing Water-Filling Score
        # For each candidate machine, compute the resulting load if this op is assigned there
        # Score by the increase in L-infinity norm (max machine load)
        machine_results = []
        for m, d in machine_options.items():
            # Current load of machine m
            current_load = state['machine_total_work'].get(m, 0.0)
            # New load if we assign this op to machine m
            new_load = current_load + d
            # Increase in max load: max(0, new_load - current_max_load)
            # But also consider the relative increase
            load_increase = max(0.0, new_load - current_max_load)
            # Relative load increase
            relative_increase = new_load / (current_max_load + 1e-9)
            # Combined cost: prefer machines that don't increase the max load much
            cost = 0.7 * (load_increase / current_max_load) + 0.3 * (relative_increase - 1.0)
            machine_results.append((cost, m, d))
        
        # Find the best machine (minimum cost)
        best_machine_result = min(machine_results, key=lambda x: x[0])
        best_machine = best_machine_result[1]
        best_machine_cost = best_machine_result[0]
        
        # Normalize machine score: lower cost is better
        # Scale to [0, 1] where 1 is best (lowest cost)
        max_cost = max(r[0] for r in machine_results)
        min_cost = min(r[0] for r in machine_results)
        if max_cost > min_cost:
            machine_score = 1.0 - (best_machine_cost - min_cost) / (max_cost - min_cost)
        else:
            machine_score = 0.5
        machine_score = max(0.0, min(1.0, machine_score))
        
        # Component 5: Flexibility bonus - more options is slightly better
        flex_score = min(num_machines / 5.0, 1.0)
        
        # Component 6: Blocking/criticality detection
        # If this op has few machine options, it's more critical to schedule
        criticality_bonus = 0.0
        if num_machines <= 1:
            criticality_bonus = 0.25
        elif num_machines == 2:
            criticality_bonus = 0.10
        
        # Near completion bonus
        if remaining_ops <= 1:
            criticality_bonus += 0.20
        
        # Component 7: Water-filling smoothness
        # Use soft-max of machine loads after assignment to measure skew
        # Lower skew is better
        if machine_results:
            # Simulate assigning to the best machine
            simulated_loads = dict(current_loads)
            simulated_loads[best_machine] = simulated_loads.get(best_machine, 0.0) + best_machine_result[2]
            soft_max_load = _soft_max(list(simulated_loads.values()), temp=0.3)
            # Normalize: lower soft-max is better
            smoothness_score = 1.0 - (soft_max_load / (current_max_load * 1.5 + 1e-9))
            smoothness_score = max(0.0, min(1.0, smoothness_score))
        else:
            smoothness_score = 0.5
        
        # Weighted combination
        w_work = 0.40
        w_ops = 0.08
        w_urgency = 0.08
        w_machine = 0.25
        w_flex = 0.05
        w_critical = 0.07
        w_smooth = 0.07
        
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_urgency * urgency_score + 
                 w_machine * machine_score + 
                 w_flex * flex_score +
                 w_critical * criticality_bonus +
                 w_smooth * smoothness_score)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
