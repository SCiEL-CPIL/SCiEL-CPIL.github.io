"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Greedy strategy: Process moduli in increasing order of m (highest weight 1/m first)
    # For each m, pick the smallest residue r that is compatible with all previous selections.
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        # Check deadline
        if time.time() > deadline:
            break
            
        # Try residues in order 0, 1, ..., m-1
        found = False
        for r in range(m):
            # Check compatibility with all previously selected (m_j, r_j)
            compatible = True
            for (m_j, r_j) in selected:
                g = gcd(m, m_j)
                if r % g == r_j % g:
                    compatible = False
                    break
            if compatible:
                selected.append((m, r))
                found = True
                break
        
        if not found:
            # No valid residue for this m; skip it
            pass
    
    # Convert to required format [[m, r], ...]
    return [[m, r] for (m, r) in selected]
# EVOLVE-BLOCK-END
