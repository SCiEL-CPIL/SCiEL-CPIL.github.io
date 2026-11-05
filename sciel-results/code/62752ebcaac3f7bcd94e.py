"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations, product
from fractions import Fraction

def construct():
    inv_float = [1.0 / (i + 1) for i in range(30)]
    best_abs = 1.0 / 870
    best_combo = (28, 29)
    best_signs = (1, -1)
    
    for k in range(1, 7):
        for combo in combinations(range(30), k):
            for signs in product([1, -1], repeat=k):
                s = 0.0
                for idx, sign in zip(combo, signs):
                    s += sign * inv_float[idx]
                if s != 0.0:
                    abs_s = abs(s)
                    if abs_s < best_abs:
                        best_abs = abs_s
                        best_combo = combo
                        best_signs = signs
    
    coeff = [0] * 30
    for idx, sign in zip(best_combo, best_signs):
        coeff[idx] = sign
    
    # Verify nonzero with exact arithmetic
    s = sum(Fraction(sign, idx + 1) for idx, sign in zip(best_combo, best_signs))
    if s == 0:
        coeff = [0] * 30
        coeff[28] = 1
        coeff[29] = -1
    
    return coeff
# EVOLVE-BLOCK-END
