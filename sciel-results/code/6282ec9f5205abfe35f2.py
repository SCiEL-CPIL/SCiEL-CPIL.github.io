import numpy as np
import time as time_module

# ------------------------------------------------------------------
#  Open-Shop dispatch rule.
#
#  Strategy: Longest Processing Time (LPT) combined with Most Remaining
#  Work (MWR) and machine load balancing.
#
#  Rationale:
#  - LPT: Scheduling operations with longer minimum processing times
#    first helps in balancing the load and preventing long jobs from
#    becoming bottlenecks late in the schedule. This is a standard
#    heuristic for minimizing makespan in flow-shop and open-shop
#    environments.
#  - MWR: Prioritizing jobs with more remaining work ensures that
#    complex jobs are not starved, which can lead to higher makespan
#    if they are left to the end.
#  - Load Balancing: Among equally prioritized operations, we prefer
#    those that go to less congested machines to avoid idle time.
# ------------------------------------------------------------------

_DEADLINE = 150.0
_START_TIME = time_module.time()
_TIME_MARGIN = 2.0  # seconds of safety margin before hard limit

# Weights for the composite score
# We prioritize LPT (min_dur) strongly, then MWR (remaining_work),
# then penalize machine congestion.
W_LPT = 1.0
W_MWR = 0.5
W_LOAD = 0.1

def _elapsed():
    return time_module.time() - _START_TIME

def _remaining_time():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _count_machine_loads(ready_operations):
    """
    Build a dict mapping machine -> count of ready ops that can use it.
    This is a proxy for machine congestion.
    """
    loads = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            loads[m] = loads.get(m, 0) + 1
    return loads

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    # Deadline guard: if we're too close to the limit, fall back to
    # a very fast single-pass heuristic immediately.
    if _remaining_time() < 0.5:
        return _fast_heuristic(ready_operations, time)
    
    # Pre-compute machine loads (congestion proxy)
    machine_loads = _count_machine_loads(ready_operations)
    
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        
        if not mo:
            # No machine options? Skip or handle gracefully. 
            # Assuming valid input, this shouldn't happen.
            continue
            
        durs = list(mo.values())
        min_dur = min(durs)
        
        # Calculate the minimum load among the available machines for this op
        # We want to assign to the least loaded machine, so the "cost" is the
        # load of that machine. Lower cost is better.
        min_load = min(machine_loads.get(m, 0) for m in mo.keys())
        
        # Composite Score:
        # 1. Maximize min_dur (LPT)
        # 2. Maximize remaining_work (MWR)
        # 3. Minimize min_load (Load Balancing)
        
        score = (W_LPT * min_dur + 
                 W_MWR * rw - 
                 W_LOAD * min_load)
        
        if score > best_score:
            best_score = score
            best_idx = i
            
        # Optional: Early exit if we are running out of time
        if _remaining_time() < 1.0:
            break
            
    return best_idx

def _fast_heuristic(ready_operations, time):
    """
    Fast single-pass heuristic used when deadline is approaching.
    Uses a simplified LPT + MWR rule without load balancing calculation
    to save time.
    """
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        rw = op.get("remaining_work", 0.0)
        mo = op.get("machine_options", {})
        
        if not mo:
            continue
            
        min_dur = min(mo.values())
        
        # Simplified score: just LPT + MWR
        score = W_LPT * min_dur + W_MWR * rw
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
