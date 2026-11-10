import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    max_mod = 128
    base_moduli = list(range(2, max_mod + 1))
    n_mods = len(base_moduli)
    
    # Precompute gcds
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = gcd(base_moduli[i], base_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    best_set = []
    best_size = 0
    
    def greedy_fill(indices_order):
        selected_indices = []
        selected_residues = {}
        
        for mod_idx in indices_order:
            m = base_moduli[mod_idx]
            valid_residues = []
            for r in range(m):
                ok = True
                for idx in selected_indices:
                    g = gcd_matrix[mod_idx][idx]
                    r_old = selected_residues[idx]
                    if r % g == r_old % g:
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                # Pick a random valid residue to diversify
                r = random.choice(valid_residues)
                selected_indices.append(mod_idx)
                selected_residues[mod_idx] = r
        
        return [(selected_residues[i], base_moduli[i]) for i in selected_indices]

    # Run random restarts
    while time.time() < deadline:
        indices = list(range(n_mods))
        random.shuffle(indices)
        
        result = greedy_fill(indices)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
            # If we reach a high number, maybe we can stop? 
            # 127 is max. 100+ is good.
            if best_size > 100:
                # Continue to see if we can improve further, but don't waste too much time
                pass

    return best_set if best_set else [[0, 2]]
