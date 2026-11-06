"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Look-ahead regret heuristic for flexible job-shop scheduling.
    
    This function selects the operation to process next by evaluating the 'regret'
    of delaying the decision for each ready operation. The regret is defined as the
    difference between the cost of the best machine assignment and the second-best
    machine assignment.
    
    The cost of assigning an operation to a machine is estimated as:
      duration + beta * estimated_congestion
    
    where estimated_congestion for a machine is the sum of durations of other
    ready operations that are also eligible for that machine. This approximates
    the immediate contention for that machine.
    
    Operations with high regret are prioritized because their machine choice is
    more critical; delaying the decision might force a suboptimal machine assignment
    due to congestion or availability changes.
    """
    if not ready_operations:
        return 0
    
    n_ops = len(ready_operations)
    
    # Pre-compute for each machine, the total duration of all ready ops eligible for it
    machine_total_load = {}
    for i, op in enumerate(ready_operations):
        for m, d in op["machine_options"].items():
            if m not in machine_total_load:
                machine_total_load[m] = 0.0
            machine_total_load[m] += d
    
    best_regret = -1.0
    best_index = 0
    best_tiebreak = (-1, 1e9, 1e9) # (remaining_work, -min_dur, -job_id) -> maximize
    
    for i, op in enumerate(ready_operations):
        options = op["machine_options"]
        if not options:
            continue
            
        # Calculate cost for each machine for this operation
        # Cost = duration + beta * (Load(m) - duration)
        # Load(m) - duration is the load from OTHER ops
        costs = []
        for m, d in options.items():
            total_load = machine_total_load.get(m, 0.0)
            other_load = total_load - d
            # Weight other load to represent congestion
            cost = d + 0.5 * other_load
            costs.append(cost)
        
        if len(costs) == 1:
            regret = 0.0
        else:
            # Sort costs to find best and second-best
            sorted_costs = sorted(costs)
            regret = sorted_costs[1] - sorted_costs[0]
            
        # Tie-breaking logic
        # 1. Maximize regret
        # 2. Maximize remaining_work (job urgency)
        # 3. Minimize min_duration (shorter op first)
        # 4. Minimize job id (deterministic)
        
        remaining_work = op["remaining_work"]
        min_dur = min(d for _, d in options.items())
        job_id = op["job"]
        
        # We want to find the op with max regret.
        # If regret is higher, it's better.
        # If regret is same, higher remaining_work is better.
        # If remaining_work is same, lower min_dur is better.
        # If min_dur is same, lower job_id is better.
        
        current_tiebreak = (remaining_work, -min_dur, -job_id)
        
        if regret > best_regret:
            best_regret = regret
            best_index = i
            best_tiebreak = current_tiebreak
        elif regret == best_regret:
            if current_tiebreak > best_tiebreak:
                best_tiebreak = current_tiebreak
                best_index = i
                
    return int(best_index)
# EVOLVE-BLOCK-END
