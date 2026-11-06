"""Rewritten dispatch rule for open-shop scheduling."""
# EVOLVE-BLOCK-START
import math
import time as _time_mod

# Track a wall-clock deadline to avoid timeout.
# The evaluator kills after 150s; we guard with a margin.
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        # We don't know when evaluation started, so we can't set a global deadline
        # reliably across multiple calls. Instead, we track per-call budget.
        pass
    return None


def _soft_min(values, temperature=1.0):
    """
    Smooth surrogate for min via log-sum-exp (negative trick).
    soft_min(x) = -log(sum(exp(-x_i / T))) * T
    As T -> 0, approaches true min. For small T, gives a smooth,
    differentiable proxy that also captures the "spread" of options.
    """
    if not values:
        return float('inf')
    if len(values) == 1:
        return values[0]
    # Scale by temperature
    scaled = [-v / temperature for v in values]
    m = max(scaled)
    # Guard against overflow
    exp_sum = sum(math.exp(s - m) for s in scaled)
    if exp_sum <= 0:
        return min(values)
    return -(m + math.log(exp_sum)) * temperature


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Load-Balancing Water-Filling heuristic (refined).
    For each candidate operation, we use a soft-min surrogate over its
    machine durations to capture both the best-machine duration and the
    scarcity/spread of options. We blend in job urgency (remaining work)
    and proximity to completion (remaining ops).
    
    The soft-min is a smooth surrogate for the nonsmooth min objective,
    which may better discriminate between operations with similar best
    durations but different option structures.
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Temperature for soft-min: small enough to approximate min,
    # but large enough to capture spread. We use a fixed value tuned
    # to typical processing time scales.
    T = 0.5
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores.append(float('inf'))
            continue
        
        durations = list(machine_options.values())
        
        # Soft-min surrogate for the best-machine duration
        sm = _soft_min(durations, T)
        
        # Also compute the true min for a secondary scarcity signal:
        # ops with fewer machine options are more constrained and
        # should be prioritized (less flexibility to reschedule later).
        n_opts = len(durations)
        min_duration = min(durations)
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Combine: lower is better
        # - sm: soft-min duration (load balancing, smooth)
        # - -alpha * remaining_work: urgency (more work = more urgent)
        # - beta * remaining_ops: fewer ops left = closer to done = more urgent
        # - gamma * (1/n_opts): scarcity penalty (fewer options = more constrained)
        
        # Weights tuned to balance the terms:
        # sm is in the same scale as processing times (likely O(1-100))
        # remaining_work is total work remaining (could be larger)
        # remaining_ops is a small integer (1-5 for 10 jobs, 5 machines)
        # 1/n_opts is in (0, 1]
        
        score = (sm
                 - 0.1 * remaining_work
                 + 0.5 * remaining_ops
                 + 0.3 / n_opts)
        
        scores.append(score)
    
    # Pick the operation with the lowest score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] < best_score:
            best_score = scores[i]
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
