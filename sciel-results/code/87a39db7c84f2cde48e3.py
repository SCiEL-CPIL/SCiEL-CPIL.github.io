"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock. We track elapsed since module load.
_T0 = _time.time()
_DEADLINE = 135.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _T0) < _DEADLINE


def _soft_min(values, temperature=0.5):
    """
    Compute a smooth approximation of the minimum using log-sum-exp trick.
    soft_min(x) = -log(sum(exp(-x_i / T))) * T
    As T -> 0, this approaches the true min.
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    
    # Shift for numerical stability
    shifted = -np.array(values) / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(log_sum * temperature)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Use a soft-min surrogate over machine durations to get a smooth
    proxy for the best machine assignment. Combine with remaining work and
    remaining ops to balance urgency with quick-completion potential.
    
    The soft-min provides a differentiable approximation of min(durations),
    which helps avoid the nonsmoothness of hard-min and may lead to better
    global behavior in the scheduling sequence.
    """
    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Gather data
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # Compute for each op: soft-min duration, mean duration, num machines
    soft_min_durs = np.zeros(n)
    mean_durs = np.zeros(n)
    num_machines = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        durs = list(op["machine_options"].values())
        if durs:
            # Use soft-min with moderate temperature for smoothness
            soft_min_durs[i] = _soft_min(durs, temperature=0.3)
            mean_durs[i] = np.mean(durs)
            num_machines[i] = len(durs)
    
    # Normalize
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1
    max_ro = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1
    max_sm = np.max(soft_min_durs) if np.max(soft_min_durs) > 0 else 1
    max_mean = np.max(mean_durs) if np.max(mean_durs) > 0 else 1
    
    # Score: we want to prioritize ops with:
    # - High remaining work (urgent) - high weight
    # - High remaining ops (complex) - moderate weight  
    # - Low soft-min duration (can finish quickly, freeing machine) - moderate weight
    # - Low mean duration (generally quick) - small weight
    
    # Weighted combination - adjusted weights to emphasize urgency more
    w_rw = 0.40   # weight for remaining work (increased)
    w_ro = 0.25   # weight for remaining ops
    w_sm = 0.25   # weight for soft-min duration (inverted: lower is better)
    w_mean = 0.10 # weight for mean duration (inverted: lower is better)
    
    score = (
        w_rw * (remaining_works / max_rw) +
        w_ro * (remaining_ops / max_ro) +
        w_sm * (1.0 - soft_min_durs / max_sm) +
        w_mean * (1.0 - mean_durs / max_mean)
    )
    
    # Pick the op with highest score
    idx = int(np.argmax(score))
    
    return idx
# EVOLVE-BLOCK-END
