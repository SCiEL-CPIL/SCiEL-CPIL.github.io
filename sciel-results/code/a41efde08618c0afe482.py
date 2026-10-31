import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Priority based on sum of squares of coordinate values.
    0^2=0, 1^2=1, 2^2=4. This non-linear weighting favors vectors
    with more 1s and 2s over 0s, breaking the symmetry of the standard
    balance heuristic and exploiting the non-linear distribution of
    affine lines in the ternary grid.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Sum of squares: 0^2=0, 1^2=1, 2^2=4
    ss = 0
    for v in vector:
        ss += v * v
    
    # Also include a small balance component to avoid degenerate solutions
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    ideal = n / 3.0
    balance = -(abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal))
    
    # Primary: sum of squares (higher = more 1s and 2s)
    # Secondary: balance (less deviation from uniform)
    return float(ss + 0.1 * balance)
