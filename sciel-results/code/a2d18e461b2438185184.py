"""Dispatch rule for open-shop scheduling using load-balanced assignment heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard (150s wall-clock limit, use ~140s as safety margin)
_DEADLINE = _time.time() + 140.0


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Model as a bipartite-like assignment where we want to minimize
    the maximum machine completion time. For each ready operation, consider
    all (operation, machine) pairs. The cost of assigning operation i to machine m
    is the resulting completion time on machine m (cumulative load + duration).
    We want to pick the (i, m) pair that minimizes this maximum completion time,
    but since we can only start ONE operation, we look for the assignment that
    gives the best balance.
    
    Practical heuristic:
    1. For each ready op, compute the best (shortest) machine option.
    2. Score each op by a composite that considers:
       - The resulting load on the chosen machine (minimize peak)
       - Remaining work (urgency)
       - Flexibility (number of machine options)
       - Shortest processing time on any machine
    3. Use a smooth surrogate (soft-min) for the max completion time objective.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline
    if _time.time() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, get machine options: list of (machine_id, duration)
    # We need to estimate current machine loads. Since we don't have direct access
    # to machine loads in the interface, we approximate using remaining_work as a proxy
    # for how much work is still to be done, and use the structure of machine_options.
    
    # Actually, in an open-shop simulation, the 'time' parameter and the fact that
    # an operation is 'ready' implies certain machines are free. The key insight:
    # we should assign to the machine that would result in the earliest completion,
    # and prioritize operations that are critical to the makespan.
    
    # Compute for each op:
    # - min_duration: shortest processing time across machines
    # - mean_duration: average processing time
    # - num_machines: flexibility
    # - best_completion_proxy: min_duration (since we don't know exact loads, use duration)
    
    min_durations = np.zeros(n)
    mean_durations = np.zeros(n)
    num_machines = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if isinstance(opts, dict):
            durations = list(opts.values())
        elif isinstance(opts, (list, tuple)):
            durations = [d if isinstance(d, (int, float)) else d.get("duration", 0) for d in opts]
        else:
            durations = [1.0]
        
        if durations:
            min_durations[i] = min(durations)
            mean_durations[i] = np.mean(durations)
            num_machines[i] = len(durations)
        else:
            min_durations[i] = 1.0
            mean_durations[i] = 1.0
            num_machines[i] = 1
    
    # Avoid division by zero
    num_machines_safe = np.maximum(num_machines, 1)
    remaining_ops_safe = np.maximum(remaining_ops, 1)
    
    # Composite score: we want to MINIMIZE a proxy for makespan contribution
    # The idea: operations with high remaining work on few machines are bottlenecks.
    # Operations with short durations reduce peak load.
    
    # Weighted score (lower is better for selection):
    # - High remaining_work -> higher priority (larger negative contribution)
    # - Short min_duration -> higher priority
    # - Few remaining_ops with high work density -> higher priority
    # - Low flexibility (few machines) -> higher priority (constrained)
    
    work_density = remaining_works / remaining_ops_safe
    flexibility_penalty = 1.0 / num_machines_safe  # Less flexible = higher penalty = higher priority
    
    # Smooth surrogate for max: use log-sum-exp style to approximate max completion
    # For each op, the "criticality" is related to how much it contributes to the makespan
    # Use a temperature-scaled soft-max for the durations
    
    temperature = 0.1
    # For each op, compute a soft-max of durations (approximates max duration smoothly)
    soft_max_durations = np.zeros(n)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if isinstance(opts, dict):
            durations = np.array(list(opts.values()), dtype=np.float64)
        elif isinstance(opts, (list, tuple)):
            durations = np.array([d if isinstance(d, (int, float)) else d.get("duration", 0) for d in opts], dtype=np.float64)
        else:
            durations = np.array([1.0])
        
        if len(durations) > 0:
            # Soft max: log(sum(exp(d/T))) * T approximates max(d)
            max_d = np.max(durations)
            shifted = (durations - max_d) / temperature
            soft_max_durations[i] = max_d + temperature * np.log(np.sum(np.exp(shifted)))
        else:
            soft_max_durations[i] = 1.0
    
    # Final composite score (higher = more urgent, select argmax)
    # Normalize components to comparable scales
    max_rw = np.max(remaining_works) if n > 0 else 1.0
    max_sd = np.max(soft_max_durations) if n > 0 else 1.0
    max_wd = np.max(work_density) if n > 0 else 1.0
    
    norm_rw = remaining_works / (max_rw + 1e-9)
    norm_sd = soft_max_durations / (max_sd + 1e-9)
    norm_wd = work_density / (max_wd + 1e-9)
    
    # Weigh: prioritize high remaining work, high work density, low flexibility,
    # and consider the soft-max duration as a proxy for machine contention
    score = (
        1.5 * norm_rw +          # Remaining work urgency
        0.8 * norm_wd +          # Work density
        0.7 * flexibility_penalty +  # Constrained ops
        0.3 * norm_sd            # Longer ops create more contention
    )
    
    idx = int(np.argmax(score))
    return idx
# EVOLVE-BLOCK-END
