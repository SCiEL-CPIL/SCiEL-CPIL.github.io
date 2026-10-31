import time

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Sum of squares of coordinate values: favors 1s and 2s over 0s
    # 0^2=0, 1^2=1, 2^2=4
    # This breaks the symmetry of the standard balance heuristic and
    # exploits the non-linear distribution of affine lines in the ternary grid
    sum_sq = sum(x * x for x in vector)
    
    # Count of 1s and 2s (non-zero entries)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Number of distinct symbols
    c0 = vector.count(0)
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    
    # Balance: prefer vectors close to uniform distribution
    ideal = n / 3.0
    balance = -(abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal))
    
    # Structural: penalize adjacent equal pairs
    struct = 0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            struct += 1
    
    # Combine: sum-of-squares as primary signal (weight 1.0),
    # distinct count as secondary (weight 0.5),
    # balance as tertiary (weight 0.3),
    # structural penalty (weight -0.2)
    return float(1.0 * sum_sq + 0.5 * distinct + 0.3 * balance - 0.2 * struct)
