"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    selected = []  # list of [m, r]
    
    # Iterate through moduli in increasing order to maximize sum of 1/m
    for m in range(20, 201):
        if time.time() > deadline:
            break
            
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
        # If no compatible residue found, skip this modulus (should be rare if any)
        
    return selected
# EVOLVE-BLOCK-END
