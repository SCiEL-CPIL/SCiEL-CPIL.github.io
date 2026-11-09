import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin
    
    # Candidates: moduli from 20 to 200
    candidates = list(range(20, 201))
    # Sort by 1/m descending (i.e., m ascending)
    candidates.sort(key=lambda m: -1.0/m)
    
    selected_moduli = []
    selected_residues = {}
    
    for m in candidates:
        if time.time() - start_time > timeout:
            break
            
        # Check compatibility with all selected moduli
        compatible = True
        for s in selected_moduli:
            if math.gcd(m, s) == 1:
                compatible = False
                break
        
        if not compatible:
            continue
            
        # Find a valid residue r for m
        valid_r = None
        for r in range(m):
            ok = True
            for s in selected_moduli:
                g = math.gcd(m, s)
                if r % g == selected_residues[s] % g:
                    ok = False
                    break
            if ok:
                valid_r = r
                break
                
        if valid_r is not None:
            selected_moduli.append(m)
            selected_residues[m] = valid_r
            
    # Format output
    classes = []
    for m in selected_moduli:
        classes.append([m, selected_residues[m]])
        
    return classes
