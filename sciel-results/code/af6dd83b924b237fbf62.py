"""Cap set greedy priority function."""
# EVOLVE-BLOCK-START
import time

_state = {
    'start_time': None,
    'deadline': None,
}

def _reset_state(n):
    """Reset state for a new n or new evaluation."""
    _state['start_time'] = time.time()
    _state['deadline'] = _state['start_time'] + 280.0  # Safety margin before 330s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    if _state['deadline'] is not None and time.time() >= _state['deadline']:
        return True
    return False

def priority(vector, n):
    """
    Assign priority to a vector in {0,1,2}^n for greedy cap set construction.
    
    Heuristic:
    1. Penalize deviation from uniform symbol distribution (quadratic).
    2. Prefer vectors with higher entropy (more balanced).
    """
    # Reset state if on first call
    if _state['start_time'] is None:
        _reset_state(n)
    
    # Deadline guard
    if _check_deadline():
        return 0.0
    
    vec = tuple(vector)
    
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vec:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        elif v == 2:
            count_2 += 1
    
    # Ideal count for each symbol is n/3
    target = n / 3.0
    
    # Quadratic penalty for deviation from target
    # This is a smooth surrogate for the "balance" objective
    dev_0 = count_0 - target
    dev_1 = count_1 - target
    dev_2 = count_2 - target
    
    deviation_score = -(dev_0 * dev_0 + dev_1 * dev_1 + dev_2 * dev_2)
    
    # Add a small bonus for vectors with exactly n/3 of each (if integer)
    # Or just use the deviation score
    
    return deviation_score
# EVOLVE-BLOCK-END
