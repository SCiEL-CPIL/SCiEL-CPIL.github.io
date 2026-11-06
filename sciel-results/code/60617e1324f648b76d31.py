"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a weighted multi-criteria heuristic.
    
    Strategy:
    - Prioritize operations with longer remaining work (to reduce makespan).
    - Prioritize operations with fewer machine options (to avoid bottlenecks).
    - Prioritize operations on machines with shorter average durations (to keep machines busy).
    - Add a small tie-breaker based on job id to ensure deterministic behavior.
    
    The score is a weighted combination:
      score = w1 * normalized_remaining_work 
            + w2 * (1 / (1 + num_machine_options))
            + w3 * (1 / (1 + avg_duration))
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    
    # Normalize remaining work to [0, 1]
    min_rw = np.min(remaining_works)
    max_rw = np.max(remaining_works)
    if max_rw > min_rw:
        norm_rw = (remaining_works - min_rw) / (max_rw - min_rw)
    else:
        norm_rw = np.ones(n, dtype=float)
    
    # Number of machine options (fewer is better for bottleneck avoidance)
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    
    # Average duration across machine options (shorter is better)
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=float)
    
    # Inverse normalize: fewer options is better, shorter duration is better
    # For num_options: map to [0, 1] where 1 = fewest options
    min_opt = np.min(num_options)
    max_opt = np.max(num_options)
    if max_opt > min_opt:
        norm_opt = 1.0 - (num_options - min_opt) / (max_opt - min_opt)
    else:
        norm_opt = np.ones(n, dtype=float)
    
    # For avg_durations: map to [0, 1] where 1 = shortest duration
    min_dur = np.min(avg_durations)
    max_dur = np.max(avg_durations)
    if max_dur > min_dur:
        norm_dur = 1.0 - (avg_durations - min_dur) / (max_dur - min_dur)
    else:
        norm_dur = np.ones(n, dtype=float)
    
    # Weighted combination
    w1 = 0.5  # remaining work
    w2 = 0.3  # machine option count
    w3 = 0.2  # average duration
    
    scores = w1 * norm_rw + w2 * norm_opt + w3 * norm_dur
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
