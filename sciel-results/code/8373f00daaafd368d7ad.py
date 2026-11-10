"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd

def construct():
    # Greedy: try to include all moduli from 20 to 200
    # Process in order of increasing m (highest weight first)
    selected = []  # list of [m, r]
    
    for m in range(20, 201):
        # Try each residue 0, 1, ..., m-1
        found = False
        for r in range(m):
            # Check compatibility with all selected classes
            compatible = True
            for m2, r2 in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        # If no residue works, skip this modulus
        # (but with 181 moduli up to 200, we should be able to include most/all)
    
    return selected
# EVOLVE-BLOCK-END
