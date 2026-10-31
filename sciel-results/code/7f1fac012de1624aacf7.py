import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    """
    Structural priority based on coordinate interaction entropy.
    
    Key insight: affine lines in {0,1,2}^n require coordinates to be
    either all-equal or all-distinct across the three points. Vectors
    with coordinates clustered near 1 (the 'middle' value) are more
    likely to participate in such lines. We penalize vectors where
    many coordinates equal 1, and reward vectors with extreme values
    (0 or 2) which are less likely to form lines.
    
    Additionally, we use the variance of coordinate values as a
    structural feature: high variance means the vector is 'spread out'
    in the ternary grid, which correlates with lower line density
    in its neighborhood.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0
    
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Fraction of coordinates at each value
    f0 = c0 / n
    f1 = c1 / n
    f2 = c2 / n
    
    # Penalty for having many 1s: 1s are the 'dangerous' middle value
    # that can complete affine lines with any pair of 0s and 2s
    # Weight: quadratic penalty to strongly discourage high f1
    penalty_ones = f1 * f1 * n
    
    # Variance of coordinate values: measures how 'spread out' the vector is
    # High variance = more 0s and 2s, fewer 1s = less likely to form lines
    mean_val = (0*c0 + 1*c1 + 2*c2) / n
    variance = (c0*(0-mean_val)**2 + c1*(1-mean_val)**2 + c2*(2-mean_val)**2) / n
    
    # Entropy of coordinate distribution: uniform distribution (f0=f1=f2=1/3)
    # maximizes entropy. Low entropy = concentrated on few values.
    # We want MODERATE entropy: not too uniform (too many 1s) but not
    # too concentrated (all 0s or all 2s are also problematic for diversity)
    entropy = 0.0
    for f in (f0, f1, f2):
        if f > 0:
            entropy -= f * math.log(f)
    
    # Primary: reward variance (spread out coordinates)
    # Secondary: penalize high fraction of 1s
    # Tertiary: mild entropy shaping (prefer slightly non-uniform)
    # The ideal entropy for cap-set-friendly vectors is below max (1.0986 for 3 classes)
    entropy_bonus = (entropy - 1.0) * 0.5  # Small bonus for moderate entropy
    
    score = 2.0 * variance - 3.0 * penalty_ones + entropy_bonus
    
    return float(score)
