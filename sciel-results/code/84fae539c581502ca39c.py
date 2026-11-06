import math
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    MAX_M = 128
    all_moduli = list(range(2, MAX_M + 1))
    
    # Precompute gcds
    gcd_table = {}
    for i in range(2, MAX_M + 1):
        for j in range(2, MAX_M + 1):
            gcd_table[(i, j)] = _gcd(i, j)
            
    best_count = 0
    best_selection = []
    
    # Run multiple trials with MRV heuristic
    for _ in range(100):
        remaining = set(all_moduli)
        selected = []
        
        while remaining:
            # Find the modulus with the fewest valid residues
            best_m = None
            best_residues = None
            min_count = float('inf')
            
            # Optimization: If we have many moduli, checking all is O(N^2).
            # N=127, so 127^2 ~ 16k operations per step, 127 steps ~ 2M ops per trial.
            # 100 trials ~ 200M ops. Might be slow in Python.
            # Let's limit MRV to a subset or use a simpler heuristic if slow.
            # Actually, 127 is small. 100 trials should be fine.
            
            for m in remaining:
                valid_residues = []
                for r in range(m):
                    valid = True
                    for r_j, m_j in selected:
                        g = gcd_table[(m, m_j)]
                        if r % g == r_j % g:
                            valid = False
                            break
                    if valid:
                        valid_residues.append(r)
                
                count = len(valid_residues)
                if count < min_count:
                    min_count = count
                    best_m = m
                    best_residues = valid_residues
                    if count == 0:
                        # Dead end, break early
                        break
                    if count == 1:
                        # Can't get better than 1
                        # But we still need to check if others are 0?
                        # If others are 0, we fail anyway.
                        # If we pick this one, we might unblock others? No, adding constraints only reduces options.
                        # So if we find 1, we can pick it immediately?
                        # Not necessarily, another modulus might have 0.
                        # So we must check all to ensure no 0s.
                        pass
            
            if best_m is None or min_count == 0:
                # No valid modulus found
                break
            
            # Pick a random valid residue
            r = random.choice(best_residues)
            selected.append([r, best_m])
            remaining.remove(best_m)
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_selection = selected
            
    return best_selection
