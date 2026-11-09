"""Open-shop dispatch rule: prioritize by remaining work with machine-load balancing."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: ensure we finish well before 120s wall-clock per evaluation.
# The dispatch function is called many times per instance; we keep overhead minimal.

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    - Primary: maximize remaining_work (jobs with more total work left are more critical).
    - Secondary: among ties, prefer the operation whose chosen machine has the least
      current/estimated load (we approximate load by preferring shorter durations on
      machines, which tends to spread work).
    - Tertiary: break ties by job id for determinism.
    
    Since we don't have direct machine load info in ready_operations, we use a heuristic:
    For each ready op, compute a score that balances:
      1) remaining_work of the job (higher = more urgent)
      2) minimum duration across available machines (shorter = less blocking)
    
    We use a weighted combination: score = remaining_work * w1 - min_duration * w2
    where w1 and w2 are tuned weights.
    
    Actually, for open-shop scheduling, a well-known good heuristic is:
    - Dispatch the operation that belongs to the job with the most remaining work.
    - Among those, pick the one with the shortest processing time on its machine.
    
    Let's implement a more nuanced approach:
    - For each ready operation, compute:
        job_remaining = op['remaining_work']
        min_dur = min(op['machine_options'].values())
        num_machines = len(op['machine_options'])
    
    - Score = job_remaining / (1 + min_dur * 0.1)  # slight penalty for long ops
    - Pick argmax.
    
    But let's think more carefully. In open-shop, the makespan is bounded by max over jobs of
    total work and max over machines of total work. A good heuristic tries to keep both balanced.
    
    A simpler and effective approach: 
    - Pick the operation whose job has the maximum remaining work.
    - Tie-break: pick the one with the smallest minimum processing time (to finish quickly and
      free up the job).
    
    Let's also consider: if we always pick max remaining_work, we might starve some machines.
    Adding a small machine-balance term could help.
    
    Revised score:
      score_i = remaining_work_i - alpha * min_duration_i
    
    where alpha is a small constant (e.g., 0.5) to slightly prefer shorter operations among
    those with similar remaining work.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        remaining_work = op['remaining_work']
        durations = list(op['machine_options'].values())
        min_dur = min(durations)
        # Primary: remaining work; secondary: prefer shorter min duration
        scores[i] = remaining_work - 0.3 * min_dur
    
    # Pick the index with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
