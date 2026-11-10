"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    elements = list(range(1, 25))
    
    def recip(x):
        return Fraction(1, x)
    
    def is_valid(S):
        S_list = list(S)
        m = len(S_list)
        for i, x in enumerate(S_list):
            target = recip(x)
            others = [S_list[j] for j in range(m) if j != i]
            # Enumerate subset sums
            sums = {Fraction(0)}
            for y in others:
                new_sums = set()
                for s in sums:
                    new_sums.add(s + recip(y))
                sums |= new_sums
            if target in sums:
                return False
        return True
    
    # Start with a good initial guess: [13,24] (12 elements)
    best = set(range(13, 25))
    best_size = len(best)
    
    # Try to improve by adding elements
    for x in range(1, 13):
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
            best_size += 1
    
    # Try more aggressive search: try all pairs to add
    improved = True
    while improved:
        improved = False
        for x in range(1, 13):
            if x not in best:
                candidate = best | {x}
                if is_valid(candidate):
                    best = candidate
                    best_size += 1
                    improved = True
    
    # Also try removing one element and adding two
    for x in list(best):
        temp = best - {x}
        for y in range(1, 25):
            if y not in temp:
                for z in range(y+1, 25):
                    if z not in temp and z != y:
                        candidate = temp | {y, z}
                        if len(candidate) > len(best) and is_valid(candidate):
                            best = candidate
                            best_size = len(candidate)
                            improved = True
    
    return sorted(best)

# EVOLVE-BLOCK-END
