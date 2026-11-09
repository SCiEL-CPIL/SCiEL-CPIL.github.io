"""Flexible Job-Shop Scheduling: dispatch rule with machine selection and heuristic scoring."""
# EVOLVE-BLOCK-START
import time as _time
import math
import random

# Global deadline guard
_DEADLINE = _time.time() + 140.0  # 10s safety margin before 150s kill


def _get_deadline_ok():
    return _time.time() < _DEADLINE


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next and optionally which machine.
    
    Strategy: For each ready operation, evaluate each eligible machine using a
    composite heuristic that considers:
      - Immediate processing time on that machine
      - Remaining work of the job (prefer jobs with more remaining work to reduce
        makespan via critical path)
      - Machine load (prefer less-loaded machines to balance)
      - Number of remaining operations (prefer jobs with fewer remaining ops to
        finish them sooner, or more remaining ops to keep them busy - we use
        a combination)
    
    We return (index, machine) to specify both operation and machine choice.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        # Pick the machine with shortest duration
        best_m = min(op["machine_options"].keys(), key=lambda m: op["machine_options"][m])
        return (0, best_m)
    
    # Estimate machine loads: we don't have direct access to machine availability,
    # but we can use the current time and durations as proxies.
    # Since we don't track machine state across calls, we use a heuristic based on
    # the operation's remaining work and duration.
    
    best_score = -float('inf')
    best_idx = 0
    best_machine = None
    
    for i, op in enumerate(ready_operations):
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0)
        machine_opts = op["machine_options"]
        
        for machine, duration in machine_opts.items():
            # Heuristic score: higher is better (we want to maximize this)
            
            # Component 1: Prefer operations with more remaining work (critical path)
            # Normalize by total work to get a relative measure
            work_score = remaining_work
            
            # Component 2: Prefer shorter processing times on the chosen machine
            # (reduces immediate blocking)
            time_penalty = -duration * 2.0
            
            # Component 3: Prefer jobs with fewer remaining operations to finish
            # them sooner (reduces idle time for downstream ops)
            # But also consider that jobs with many remaining ops need to stay busy.
            # Use a balance: slightly prefer fewer remaining ops for completion.
            ops_penalty = -remaining_ops * 0.5
            
            # Component 4: Prefer machines that are likely less loaded.
            # We don't have direct machine load info, so use duration as proxy:
            # shorter duration means less time the machine is occupied.
            # This is already captured in time_penalty.
            
            # Component 5: Prefer operations where the chosen machine has the
            # shortest duration relative to other options (efficiency)
            min_dur = min(machine_opts.values())
            max_dur = max(machine_opts.values())
            if max_dur > min_dur:
                efficiency = (max_dur - duration) / (max_dur - min_dur)  # 1.0 if best, 0.0 if worst
            else:
                efficiency = 1.0
            eff_bonus = efficiency * duration * 0.3  # Scale by duration to make it meaningful
            
            score = work_score + time_penalty + ops_penalty + eff_bonus
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = machine
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
