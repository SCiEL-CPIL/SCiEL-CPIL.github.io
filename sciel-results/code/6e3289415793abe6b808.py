def construct():
    from math import lcm
    from fractions import Fraction

    # We want to maximize sum of 1/x.
    # Hypothesis: Optimal set contains at most one element from [2, 128].
    # Proof sketch: If x, z in [2, 128] are both in S, lcm(x,z) > 256.
    # If x=2, no z in [3,128] is compatible.
    # If x is odd, z must be even? lcm(x,z) = xz (if coprime).
    # It is possible to have multiple, but small numbers dominate.
    # Let's just evaluate the single-anchor strategy and the empty-anchor strategy.
    # If we find a case where two anchors are better, we can extend, but 2 is so dominant.
    
    best_score = Fraction(0)
    best_set = []
    
    # Case 0: No element from [2, 128]
    # Set is all y in [129, 256]
    score_0 = Fraction(0)
    set_0 = []
    for y in range(129, 257):
        score_0 += Fraction(1, y)
        set_0.append(y)
    if score_0 > best_score:
        best_score = score_0
        best_set = set_0
        
    # Case 1: Exactly one element x from [2, 128]
    for x in range(2, 129):
        score = Fraction(1, x)
        current_set = [x]
        for y in range(129, 257):
            if lcm(x, y) > 256:
                score += Fraction(1, y)
                current_set.append(y)
        if score > best_score:
            best_score = score
            best_set = current_set
            
    return best_set
