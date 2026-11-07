import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_VAL = 1000000
    
    # Generate critical points where the state (set of multiples) changes
    # State changes when S crosses a multiple of any divisor.
    # We collect all multiples of each divisor up to MAX_VAL.
    critical_points = set()
    for d in divisors:
        # Multiples: d, 2d, 3d, ... <= MAX_VAL
        # The state for S is determined by ceil(S/d).
        # ceil(S/d) = k for S in ((k-1)d, kd].
        # So the state is constant on intervals ending at k*d.
        # We should check S = k*d for all k.
        for k in range(1, MAX_VAL // d + 1):
            critical_points.add(k * d)
            
    # Add 1 to ensure we cover the start
    critical_points.add(1)
    
    # Sort the critical points
    points = sorted(critical_points)
    
    best_range = float('inf')
    best_vals = None
    
    for S in points:
        # Calculate the multiples for this S
        vals = []
        valid = True
        for d in divisors:
            # ceil(S/d) * d
            # If S is divisible by d, ceil(S/d) = S/d
            # If not, ceil(S/d) = (S//d) + 1
            k = (S + d - 1) // d
            v = k * d
            if v > MAX_VAL:
                valid = False
                break
            vals.append(v)
        
        if not valid:
            continue
            
        # Check distinctness
        if len(set(vals)) != n:
            continue
            
        # Check positive (already ensured by k>=1 since S>=1, d>=1)
        # Check bounds (already ensured)
        
        current_range = max(vals) - min(vals)
        if current_range < best_range:
            best_range = current_range
            best_vals = vals[:]
            
    if best_vals is None:
        # Fallback, though should not happen
        return [d for d in divisors]
        
    return [int(v) for v in best_vals]
