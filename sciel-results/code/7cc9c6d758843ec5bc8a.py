import random
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    moduli = list(range(20, 201))
    # Sort by weight descending (smallest m first)
    moduli.sort(key=lambda m: -1.0/m)
    
    best_score = 0.0
    best_set = []
    
    # Precompute gcds
    gcds = {}
    for i, m1 in enumerate(moduli):
        for m2 in moduli[i+1:]:
            g = gcd(m1, m2)
            gcds[(m1, m2)] = g
            gcds[(m2, m1)] = g
    
    # Run multiple random restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        selected = []
        selected_mods = set()
        
        for m in moduli:
            # Check coprime constraint (must share factor with all selected)
            compatible = True
            for m2, r2 in selected:
                if gcds[(m, m2)] == 1:
                    compatible = False
                    break
            if not compatible:
                continue
            
            # Find valid residue using blocked array
            blocked = [False] * m
            for m2, r2 in selected:
                g = gcds[(m, m2)]
                # r % g == r2 % g
                remainder = r2 % g
                for r in range(remainder, m, g):
                    blocked[r] = True
            
            valid_residues = [r for r in range(m) if not blocked[r]]
            
            if valid_residues:
                # Pick one randomly to diversify
                r = random.choice(valid_residues)
                selected.append([m, r])
                selected_mods.add(m)
            
        # Calculate score
        score = sum(1.0/m for m, r in selected)
        if score > best_score:
            best_score = score
            best_set = selected[:]
            
    return best_set
