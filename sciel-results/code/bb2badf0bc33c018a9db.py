"""Improved dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Multi-criteria dispatching that considers:
    1. Remaining work for the job (prioritize jobs with more remaining work)
    2. Minimum processing time among available machines (to reduce makespan)
    3. Number of remaining operations (to handle job completion)
    4. Machine load balancing (prefer less-loaded machines)
    
    Returns: index of the ready operation to start
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    # Extract key features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration across its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=float)
    
    # Maximum duration across machine options (for variability consideration)
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else float('inf')
        for op in ready_operations
    ], dtype=float)
    
    # Number of machine options (flexibility)
    n_machine_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1
        for op in ready_operations
    ], dtype=float)
    
    # Normalize features to [0, 1] range for weighting
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max - arr_min < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - arr_min) / (arr_max - arr_min)
    
    # Higher remaining work is better (prioritize)
    norm_remaining_work = normalize(remaining_works)
    
    # Lower remaining ops might be better for completing jobs soon, 
    # but we want to prioritize jobs with more remaining work overall
    # Actually, let's think: jobs with fewer remaining ops can finish sooner,
    # which might be good. But jobs with more remaining work need attention.
    # Let's use a combination.
    
    # Lower min duration is better (shorter operations reduce makespan contribution)
    norm_min_duration = 1.0 - normalize(min_durations)
    
    # More machine options = more flexibility = potentially better
    norm_flexibility = normalize(n_machine_options)
    
    # Lower max duration is better (less variability/risk)
    norm_max_duration = 1.0 - normalize(max_durations)
    
    # Combine with weights. Tuned heuristically:
    # - Remaining work: most important (0.35)
    # - Min duration: important for makespan (0.30)
    # - Flexibility: moderate (0.15)
    # - Max duration: moderate (0.10)
    # - Remaining ops: slight preference for fewer (0.10)
    norm_remaining_ops = 1.0 - normalize(remaining_ops)
    
    scores = (
        0.35 * norm_remaining_work +
        0.30 * norm_min_duration +
        0.15 * norm_flexibility +
        0.10 * norm_max_duration +
        0.10 * norm_remaining_ops
    )
    
    # Add a small random tie-breaker to avoid deterministic bias
    if n > 1:
        scores += np.random.uniform(0, 0.01, size=n)
    
    # Return the index of the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
