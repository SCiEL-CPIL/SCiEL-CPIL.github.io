import numpy as np
import math

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    # Precompute machine loads if possible (track which machines are likely busy)
    # Since we don't have direct access to machine schedules, we use a heuristic:
    # Assume machines are equally loaded on average, but prefer machines with shorter durations

    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 1)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        # Use soft-min to smoothly select machine: lower temperature = closer to hard min
        # Temperature of 1.0 gives a smooth preference for shorter durations
        temps = list(machine_options.values())
        if not temps:
            continue
        
        # Soft-min: -log(sum(exp(-d/T))) / T, but we want to minimize duration
        # Use: score_machine = -log(sum(exp(-d/T))) for each machine, pick best
        # Simpler: use weighted average with exponential weights favoring shorter times
        T = 2.0  # Temperature for soft-min
        exp_weights = [math.exp(-d / T) for d in temps]
        total_weight = sum(exp_weights)
        if total_weight < 1e-15:
            # Fallback to hard min
            min_idx = temps.index(min(temps))
            best_m_for_op = list(machine_options.keys())[min_idx]
            min_duration = temps[min_idx]
        else:
            # Expected duration under soft-min distribution
            min_duration = sum(d * w for d, w in zip(temps, exp_weights)) / total_weight
            # Pick the machine that minimizes duration (hard min for actual assignment)
            min_idx = temps.index(min(temps))
            best_m_for_op = list(machine_options.keys())[min_idx]
        
        if min_duration <= 0:
            min_duration = 1e-9
            
        # Composite heuristic:
        # 1. Urgency density: remaining_work / processing_time (higher = more urgent)
        # 2. Completion proximity: 1/remaining_ops (higher = closer to done)
        # 3. Machine efficiency: prefer machines where this op is relatively fast
        
        urgency = remaining_work / (min_duration + 1e-9)
        completion_proximity = 1.0 / (remaining_ops + 1e-9)
        
        # Weighted combination: urgency dominates, completion as tiebreaker
        score = 0.85 * urgency + 0.15 * completion_proximity * (remaining_work + 1) / (min_duration + 1)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op
    
    # If no valid operation found, fallback to first
    if best_machine is None:
        return 0
    
    return (best_index, best_machine)
