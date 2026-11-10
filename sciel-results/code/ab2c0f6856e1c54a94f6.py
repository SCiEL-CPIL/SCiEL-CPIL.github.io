import random
from math import gcd

def construct():
        moduli = list(range(20, 201))
        # Sort by weight descending (smallest m first)
        moduli.sort(key=lambda m: -1.0/m)
        
        best_score = 0.0
        best_set = []
        
        # Run multiple random restarts
        for _ in range(1000):
            selected = []
            # Shuffle slightly to break ties? Or just use fixed order.
            # Fixed order is deterministic. Randomization in residue choice is key.
            
            for m in moduli:
                # Check coprime constraint
                compatible = True
                for m2, r2 in selected:
                    if gcd(m, m2) == 1:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Find valid residue
                valid_residues = []
                for r in range(m):
                    ok = True
                    for m2, r2 in selected:
                        g = gcd(m, m2)
                        if r % g == r2 % g:
                            ok = False
                            break
                    if ok:
                        valid_residues.append(r)
                
                if valid_residues:
                    # Pick one randomly to diversify
                    r = random.choice(valid_residues)
                    selected.append([m, r])
            
            # Calculate score
            score = sum(1.0/m for m, r in selected)
            if score > best_score:
                best_score = score
                best_set = selected[:]
        
        return best_set
