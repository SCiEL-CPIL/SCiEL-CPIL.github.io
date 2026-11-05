import math
import time

def construct():
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    # Greedy: try moduli in increasing order (smaller m gives larger 1/m)
    # For each m, find a residue r in [0, m) such that for all selected (m2, r2):
    #   r % gcd(m, m2) != r2 % gcd(m, m2)
    
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
        
        # Find a valid residue for modulus m
        # For each existing (m2, r2), we forbid r ≡ r2 (mod g) where g = gcd(m, m2)
        # We need r in [0, m) not in any forbidden class
        
        best_r = None
        # Try residues in order 0, 1, 2, ... (prefer small residues for consistency)
        for r in range(m):
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                best_r = r
                break
        
        if best_r is not None:
            selected.append((m, best_r))
    
    # Convert to required format
    return [[m, r] for m, r in selected]
