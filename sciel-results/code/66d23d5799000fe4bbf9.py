"""Optimized dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math
from collections import defaultdict
from heapq import heappush, heappop, heapify

# Global state for the simulator (per evaluation run)
_sim_state = {}
_DEADLINE = 115.0  # Safety margin before 120s limit
_start_time = None
_best_score = None
_best_dispatch_fn = None

def _reset_sim_state():
    global _sim_state, _start_time, _best_score, _best_dispatch_fn
    _sim_state = {}
    _start_time = time.time()
    _best_score = float('inf')
    _best_dispatch_fn = None

def _check_time():
    if _start_time is None:
        return False
    return (time.time() - _start_time) > _DEADLINE

def _simulate(ready_ops_init, dispatch_fn, time_budget=5.0):
    """
    Simulate the open-shop schedule using the given dispatch function.
    ready_ops_init: list of dicts as provided by the evaluator
    dispatch_fn: function (ready_ops, time) -> index
    Returns: makespan
    """
    # We need to reconstruct the full problem state from ready_ops_init
    # Since we only get ready_operations at each call, we need to maintain state
    # But for offline optimization, we'll simulate based on the initial state
    
    # Actually, for the MODEL-AND-OPTIMIZE approach, we need a different strategy.
    # The dispatch function is called at each event. We can't easily simulate offline
    # without the full problem instance.
    
    # Alternative: use a heuristic that is well-known to perform well for open-shop.
    # Common good heuristics for open-shop:
    # 1. Most Work Remaining (MWR) - current
    # 2. Longest Processing Time (LPT) on the operation
    # 3. Most Operations Remaining (MOR)
    # 4. A combination: weighted sum of remaining work, operations remaining, etc.
    
    # Let's implement a parameterized heuristic and tune parameters.
    pass

def _make_dispatch_fn(params):
    """Create a dispatch function from parameters."""
    def dispatch(ready_operations, time):
        if not ready_operations:
            return 0
        
        # Check if we should fall back to a simple rule due to time pressure
        if _check_time():
            # Fall back to most work remaining
            return max(range(len(ready_operations)), 
                      key=lambda i: ready_operations[i]["remaining_work"])
        
        w_work = params.get("w_work", 1.0)
        w_ops = params.get("w_ops", 0.5)
        w_lpt = params.get("w_lpt", 0.0)
        w_machine_load = params.get("w_machine_load", 0.0)
        w_tiebreak = params.get("w_tiebreak", 0.1)
        
        scores = []
        for i, op in enumerate(ready_operations):
            remaining_work = op["remaining_work"]
            remaining_ops = op["remaining_ops"]
            
            # LPT: longest processing time among machine options
            machine_options = op["machine_options"]
            if machine_options:
                max_duration = max(machine_options.values())
                min_duration = min(machine_options.values())
            else:
                max_duration = 0
                min_duration = 0
            
            # Machine load: estimate based on which machines are busy
            # We don't have direct access to machine load in the interface,
            # so we approximate using the machine options
            # Actually, we can't know machine load from the interface alone.
            # Let's use a different feature: the ratio of min to max duration
            flexibility = (min_duration / max_duration) if max_duration > 0 else 1.0
            
            score = (w_work * remaining_work + 
                     w_ops * remaining_ops * 10 +  # Scale ops to comparable range
                     w_lpt * max_duration +
                     w_tiebreak * flexibility)
            
            scores.append(score)
        
        # Pick the operation with the highest score
        best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
        return int(best_idx)
    
    return dispatch

def _evaluate_params(params, sample_instances=None):
    """
    Evaluate parameters by running the dispatch on the current instance.
    Since we don't have access to the full instance offline, we use a proxy metric.
    """
    # For now, we'll rely on known good heuristics from literature.
    # The open-shop problem benefits from:
    # - Prioritizing jobs with most remaining work
    # - Considering the number of remaining operations
    # - Preferring operations that can be processed quickly (less blocking)
    
    # This is a placeholder for actual evaluation
    return 0.0

def dispatch(ready_operations, time):
    """
    Dispatch rule that selects which ready operation to start next.
    
    Uses a combined MWR job selection with SPT machine selection strategy
    for open-shop scheduling. Returns (index, machine) tuple.
    """
    # Initialize global state on first call
    if _start_time is None:
        _reset_sim_state()
    
    if not ready_operations:
        return (0, None)
    
    # Time guard: if we're running out of time, use a fast simple rule
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
    
    # Combined heuristic:
    # - Primary: Most Work Remaining (MWR) for job selection
    # - Secondary: Shortest Processing Time (SPT) for machine selection
    # - Tertiary: Consider remaining operations and flexibility
    
    w_work = 1.0
    w_ops = 0.2
    w_spt = 0.15
    w_flex = 0.05
    
    scores = []
    best_machines = []
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations)
            max_duration = max(durations)
            avg_duration = sum(durations) / len(durations)
            # Flexibility: how many machines can do this op
            num_machines = len(machine_options)
            # Choose SPT machine
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
        else:
            min_duration = 0
            max_duration = 0
            avg_duration = 0
            num_machines = 1
            best_machine = None
        
        # Score components:
        # 1. Remaining work (primary)
        work_score = remaining_work
        
        # 2. Remaining operations (secondary)
        ops_score = remaining_ops * 3.0
        
        # 3. SPT preference: shorter min duration is better
        # Normalize to [0,1] range by using inverse
        spt_score = 1.0 / (min_duration + 1.0) if min_duration > 0 else 1.0
        
        # 4. Flexibility: more machine options is better
        flex_score = min(num_machines / 5.0, 1.0)
        
        score = (w_work * work_score + 
                 w_ops * ops_score + 
                 w_spt * spt_score * 10 +  # Scale SPT to be meaningful
                 w_flex * flex_score * 5)
        
        scores.append(score)
        best_machines.append(best_machine)
    
    best_idx = max(range(len(ready_operations)), key=lambda i: scores[i])
    return (best_idx, best_machines[best_idx])

# EVOLVE-BLOCK-END
