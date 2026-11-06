def construct():
    from fractions import Fraction
    
    # Start with all numbers in (128, 256]
    S = set(range(129, 257))
    
    def reciprocal_sum(S):
        return sum(Fraction(1, x) for x in S)
    
    current_sum = reciprocal_sum(S)
    
    # Greedily add small numbers if beneficial
    improved = True
    while improved:
        improved = False
        for x in range(2, 129):
            if x in S:
                continue
            # Incompatible: y in S where y % x == 0 (since y > x)
            incompatible = [y for y in S if y % x == 0]
            if not incompatible:
                # Can add without removing anything
                S.add(x)
                current_sum += Fraction(1, x)
                improved = True
            else:
                gain = Fraction(1, x) - sum(Fraction(1, y) for y in incompatible)
                if gain > 0:
                    for y in incompatible:
                        S.discard(y)
                    S.add(x)
                    current_sum += gain
                    improved = True
    
    return list(S)
