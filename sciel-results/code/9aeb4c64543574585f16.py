# EVOLVE-BLOCK-START
import random
from math import gcd

def construct():
    best_score = 0.0
    best_selection = []
    
    # Number of restarts. 500 seems reasonable for speed.
    for _ in range(500):
        selected = []
        valid_restart = True
        
        # Process moduli in increasing order (highest weight first)
        for m in range(20, 201):
            # Create a list of residues and shuffle
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
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
            
            if not found:
                # If we can't place a residue for a small modulus, 
                # this restart is likely suboptimal.
                if m < 100:
                    valid_restart = False
                    break
                # For larger moduli, we just skip and continue
                # (The weight 1/m is small, so losing one is less critical)
        
        if valid_restart:
            # Calculate score
            score = sum(1.0 / m for m, r in selected)
            if score > best_score:
                best_score = score
                best_selection = selected
            
            # If we included all moduli, we can't do better
            if len(selected) == 181:
                break

    # Fallback if no valid restart found (should not happen)
    if not best_selection:
        # Try a deterministic greedy as fallback
        selected = []
        for m in range(20, 201):
            for r in range(m):
                compatible = True
                for m2, r2 in selected:
                    g = gcd(m, m2)
                    if r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    selected.append([m, r])
                    break
        best_selection = selected

    return best_selection
# EVOLVE-BLOCK-END
