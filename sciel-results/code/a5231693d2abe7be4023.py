"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for evaluator's 150s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # Sort moduli by 1/m (i.e., ascending m) to prioritize smaller moduli
    moduli_sorted = sorted(moduli, key=lambda m: 1.0/m, reverse=True)
    
    # Greedy construction: pick moduli in order of 1/m, assign best residue
    selected = []  # list of (m, r)
    selected_moduli = set()
    
    for m in moduli_sorted:
        if time.time() > deadline - 5:
            break
            
        # Check if m is already selected
        if m in selected_moduli:
            continue
            
        # Find best residue for m that is compatible with all selected
        best_r = None
        for r in range(m):
            compatible = True
            for (m2, r2) in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                best_r = r
                break
        
        if best_r is not None:
            selected.append((m, best_r))
            selected_moduli.add(m)
    
    # Convert to required format
    result = [[m, r] for m, r in selected]
    return result
# EVOLVE-BLOCK-END
