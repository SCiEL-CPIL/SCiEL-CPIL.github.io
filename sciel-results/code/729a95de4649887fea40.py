import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    moduli = list(range(2, 129))
    best_result = []
    
    # Precompute gcds? No, just use math.gcd, it's fast.
    
    while time.time() - start_time < timeout:
        random.shuffle(moduli)
        current_list = []
        
        for m in moduli:
            found = False
            # Try residues in random order to break ties?
            # Or just 0 to m-1.
            # Random order of residues might help if one specific residue is blocking.
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                compatible = True
                for r_i, m_i in current_list:
                    g = math.gcd(m, m_i)
                    if (r - r_i) % g == 0:
                        compatible = False
                        break
                if compatible:
                    current_list.append([r, m])
                    found = True
                    break
            
            # If not found, we just skip m in this trial
            
        if len(current_list) > len(best_result):
            best_result = current_list
            # If we found all 127, we can stop
            if len(best_result) == 127:
                break
                
    return best_result
