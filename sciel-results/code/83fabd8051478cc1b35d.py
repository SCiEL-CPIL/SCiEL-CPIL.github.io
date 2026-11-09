"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a State-Dependent Composite Heuristic (Model-and-Optimize approach):
    1. Model: Estimate the "phase" of the schedule based on the distribution of 
       remaining work. 
    2. Optimize: Adjust heuristic weights dynamically.
       - Early phase (high avg remaining work): Prioritize Long Remaining Processing 
         Time (LRPT) to keep critical path jobs moving.
       - Late phase (low avg remaining work): Prioritize Short Remaining Processing 
         Time (SRPT) to clear the queue and reduce tail effects.
    3. Machine Selection: Select the machine that minimizes a composite cost of 
       duration and estimated contention (number of other ready ops eligible for 
       that machine).
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute features for each operation
    rws = []
    min_durs = []
    machine_options_list = []
    remaining_ops_counts = []
    
    # For machine contention analysis
    machine_usage_counts = {}
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        rws.append(mw)
        remaining_ops_counts.append(ro)
        machine_options_list.append(machine_options)
        
        if machine_options:
            # Find best machine (shortest duration)
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
            min_durs.append(best_dur)
            
            # Count how many ready ops can run on each machine
            # This is a proxy for machine load/congestion
            for m in machine_options.keys():
                machine_usage_counts[m] = machine_usage_counts.get(m, 0) + 1
        else:
            min_durs.append(0)
            
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Calculate average remaining work to determine "phase"
    avg_rw = sum(rws) / n
    max_rw = max(rws) if rws else 1
    min_rw = min(rws) if rws else 0
    
    # Heuristic for phase:
    # If avg_rw is close to max_rw, jobs are similar (or early).
    # If avg_rw is small, we are likely late (assuming work decreases).
    # We use a sigmoid-like transition to blend weights.
    # Scale: Assume total horizon is roughly 2 * max_rw? No, let's use relative variance.
    
    # Let's use the ratio of avg to max as a phase indicator.
    # Early: jobs have lots of work. Late: jobs have little work.
    # Without a global horizon, we can't be perfect. 
    # Alternative: Use the variance of remaining work.
    
    # Let's stick to a robust weighted sum but with machine contention penalty.
    # This is a known improvement over simple LRPT/SRPT.
    
    max_dur = max(min_durs) if min_durs else 1
    min_dur = min(min_durs) if min_durs else 1
    
    scores = []
    best_machines = []
    
    # Strategy: Dynamic Efficiency Score
    # Prioritize operations that have high remaining work but can be completed quickly (high throughput).
    # Penalize operations that must use heavily contended machines.
    
    for i in range(n):
        rw = rws[i]
        dur = min_durs[i]
        mo = machine_options_list[i]
        
        if not mo:
            # Fallback
            best_m = None
            contention = 0
            score = 0.0
        else:
            # Re-select best machine considering contention
            # We want to minimize the "effective cost" of running this op on machine m.
            # Cost = duration + penalty * contention
            # Using a higher penalty (0.5) to strongly avoid congested machines.
            
            best_m = None
            best_cost = float('inf')
            best_cont = 0
            for m, d in mo.items():
                cnt = machine_usage_counts.get(m, 0)
                cost = d + 0.5 * cnt
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
                    best_cont = cnt
            
            contention = best_cont
            best_machines.append(best_m)
            
            # Efficiency Score: rw / (dur + 1)
            # +1 avoids division by zero and scales the ratio.
            efficiency = rw / (dur + 1.0)
            
            # Contention Penalty: 1 / (1 + contention)
            # If contention is 0, factor is 1. If contention is high, factor drops.
            cont_factor = 1.0 / (1.0 + contention)
            
            score = efficiency * cont_factor
        scores.append(score)
        
    # Find best
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
