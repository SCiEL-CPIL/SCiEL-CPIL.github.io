"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Dispatch rule using a composite heuristic that considers:
    1. Critical path length (remaining work + downstream dependencies)
    2. Machine load balance (prefer less busy machines)
    3. Operation duration on preferred machine
    
    This is a more sophisticated rule than simple longest remaining work.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    n = len(ready_operations)
    
    # Calculate a composite score for each ready operation
    # We want to maximize this score (pick the operation with highest score)
    
    # First, estimate machine loads based on what's currently scheduled
    # Since we don't have direct access to machine loads, we use heuristics
    
    scores = np.zeros(n)
    
    # Get the minimum duration for each operation across its machine options
    min_durations = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
    
    min_durations = np.array(min_durations)
    
    # Remaining work
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    
    # Remaining operations count (proxy for critical path depth)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Normalize components to [0, 1]
    def normalize(arr):
        if arr.max() - arr.min() < 1e-9:
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / (arr.max() - arr.min())
    
    norm_work = normalize(remaining_works)
    norm_ops = normalize(remaining_ops)
    norm_dur = normalize(min_durations)
    
    # Composite score:
    # - Higher remaining work is more urgent (weight: 0.5)
    # - More remaining operations means longer critical path (weight: 0.3)
    # - Shorter duration on best machine allows earlier completion (weight: 0.2, inverted)
    
    # For duration, we prefer shorter (so invert)
    score = 0.5 * norm_work + 0.3 * norm_ops + 0.2 * (1 - norm_dur)
    
    # Add a small tie-breaker based on job number to ensure deterministic behavior
    # and break ties in a way that might help balance load
    job_ids = np.array([op["job"] for op in ready_operations])
    if job_ids.max() - job_ids.min() > 0:
        tie_breaker = (job_ids - job_ids.min()) / (job_ids.max() - job_ids.min())
        score += 0.01 * tie_breaker
    else:
        score += 0.01 * np.linspace(0, 1, n)
    
    return int(np.argmax(score))
# EVOLVE-BLOCK-END
