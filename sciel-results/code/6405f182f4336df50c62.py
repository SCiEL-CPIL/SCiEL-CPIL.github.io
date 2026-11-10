from fractions import Fraction

def construct():
    # We want to maximize size of S subset of 1..24
    # Strategy:
    # 1. Greedily build S_large from 24 down to 2.
    # 2. Check if we can add 1.
    
    sums = {Fraction(0)}
    selected = []
    
    # Iterate from 24 down to 2
    for k in range(24, 1, -1):
        target = Fraction(1, k)
        if target in sums:
            # Conflict, skip k
            continue
        else:
            # Include k
            selected.append(k)
            # Update sums
            new_sums = set(sums)
            for s in sums:
                ns = s + target
                if ns <= 1: # Prune sums > 1
                    new_sums.add(ns)
            sums = new_sums
            
    # Now check if we can add 1
    # Condition: 1 is not in sums (sums of reciprocals of selected)
    if Fraction(1) not in sums:
        selected.append(1)
        
    return selected
