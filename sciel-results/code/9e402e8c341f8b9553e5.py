import math
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Greedy approach: process moduli in increasing order (smaller m = larger weight 1/m)
    # For each m, find a residue r that is compatible with all previously selected pairs.
    # Compatibility: r % gcd(m, m2) != r2 % gcd(m, m2) for all selected (m2, r2)
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Find a valid residue for modulus m
        # For each existing (m2, r2), we forbid r ≡ r2 (mod g) where g = gcd(m, m2)
        # We need r in [0, m) not in any forbidden class
        
        best_r = None
        for r in range(m):
            compatible = True
            for (m2, r2) in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                best_r = r
                break
        
        if best_r is not None:
            selected.append((m, best_r))
    
    # Try to improve by removing some selections and trying different residues
    # This is a local search / hill climbing step
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i] and re-adding it with a different residue
            m_i, r_i = selected[i]
            saved = selected.pop(i)
            
            # Try all residues for m_i
            for r in range(m_i):
                compatible = True
                for j, (m2, r2) in enumerate(selected):
                    if j == i:
                        continue
                    g = math.gcd(m_i, m2)
                    if r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    if r != r_i:
                        selected.insert(i, (m_i, r))
                        improved = True
                    else:
                        selected.insert(i, saved)
                    break
            else:
                # No alternative found, put it back
                selected.insert(i, saved)
        
        # Also try swapping: remove one element and see if we can add a different modulus
        # This is more complex, so we'll stick with the simple improvement for now
    
    # Convert to the required format
    return [[m, r] for (m, r) in selected]
