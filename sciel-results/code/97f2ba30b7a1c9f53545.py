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
    
    # Phase indicator: ratio of avg to max remaining work
    # Early phase: high ratio (jobs similar, lots of work)
    # Late phase: low ratio (jobs diverged, little work)
    phase_ratio = avg_rw / (max_rw + 1e-9) if max_rw > 0 else 1.0
    
    scores = []
    best_machines = []
    
    # Strategy: Dynamic Bottleneck Optimization
    # Score = (remaining_work / soft_min_duration) * bottleneck_factor * contention_factor
    # 
    # soft_min_duration: Smooth surrogate for min duration using log-sum-exp
    #   Captures machine option diversity - ops with multiple fast machines score higher
    # bottleneck_factor: Ratio of job's remaining work to total eligible machine load
    #   Prioritizes jobs that are "bottlenecked" (high work relative to available capacity)
    # contention_factor: Smooth penalty for machine congestion
    
    # Pre-compute total load per machine (sum of min durations of ops eligible for that machine)
    machine_total_load = {}
    for i in range(n):
        mo = machine_options_list[i]
        if mo:
            for m in mo.keys():
                machine_total_load[m] = machine_total_load.get(m, 0) + min_durs[i]
    
    for i in range(n):
        rw = rws[i]
        mo = machine_options_list[i]
        
        if not mo:
            best_m = None
            score = 0.0
        else:
            # Select best machine using smooth cost: duration + smooth contention penalty
            best_m = None
            best_cost = float('inf')
            best_cont = 0
            for m, d in mo.items():
                cnt = machine_usage_counts.get(m, 0)
                # Smooth contention penalty: use tanh-like smooth function
                # For small cnt, penalty ~ cnt; for large cnt, saturates
                smooth_pen = 0.3 * cnt / (1.0 + 0.5 * cnt)
                cost = d + smooth_pen
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
                    best_cont = cnt
            
            best_machines.append(best_m)
            
            # Soft-min duration: log-sum-exp based smooth minimum
            # For durations [d1, d2, ...], soft_min = -log(sum(exp(-di/T))) * T
            # With T=1.0, this gives a smooth approximation to min
            durs_list = list(mo.values())
            if len(durs_list) == 1:
                soft_min_dur = durs_list[0]
            else:
                # Soft min via log-sum-exp with temperature
                T = 1.0
                neg_durs = [-d / T for d in durs_list]
                max_neg = max(neg_durs)
                sum_exp = sum(__import__('math').exp(x - max_neg) for x in neg_durs)
                soft_min_dur = -(max_neg + __import__('math').log(sum_exp)) * T
                soft_min_dur = max(soft_min_dur, 1e-9)
            
            # Efficiency: remaining work per unit time
            efficiency = rw / (soft_min_dur + 1.0)
            
            # Bottleneck factor: how "critical" this job is relative to machine capacity
            # Total eligible machine load for this op
            total_eligible_load = sum(machine_total_load.get(m, 0) for m in mo.keys())
            if total_eligible_load > 0:
                bottleneck_factor = rw / (total_eligible_load + 1.0)
            else:
                bottleneck_factor = 1.0
            
            # Contention factor: smooth penalty
            cont_factor = 1.0 / (1.0 + 0.7 * best_cont)
            
            # Phase-adaptive weighting:
            # Early phase: weight efficiency more (keep critical jobs moving)
            # Late phase: weight bottleneck more (clear congested paths)
            w_eff = 0.6 + 0.4 * (1.0 - phase_ratio)
            w_bott = 0.4 + 0.4 * phase_ratio
            
            score = (w_eff * efficiency + w_bott * bottleneck_factor) * cont_factor
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
