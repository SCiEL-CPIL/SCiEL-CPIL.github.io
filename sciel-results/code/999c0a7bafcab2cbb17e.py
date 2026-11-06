import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Candidate moduli: 2 to 128
    all_moduli = list(range(2, 129))
    n_mods = len(all_moduli)
    
    # Precompute gcd matrix for speed
    # gcd_matrix[i][j] = gcd(all_moduli[i], all_moduli[j])
    gcd_matrix = [[0]*n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_matrix[i][j] = math.gcd(all_moduli[i], all_moduli[j])
            
    # Map modulus value to index for faster lookup? 
    # Or just use indices.
    # We will work with indices of all_moduli.
    
    def is_disjoint_idx(r1, i1, r2, i2):
        g = gcd_matrix[i1][i2]
        if g == 1:
            return False
        return (r1 - r2) % g != 0

    def can_add_idx(r, i, selected_indices, selected_residues):
        for k in range(len(selected_indices)):
            j = selected_indices[k]
            r_j = selected_residues[k]
            g = gcd_matrix[i][j]
            if g == 1:
                return False
            if (r - r_j) % g == 0:
                return False
        return True

    best_set = []
    best_len = 0
    
    # Number of restarts
    # Let's try 200 restarts, adjust if time allows
    # We will run until time limit
    
    while time.time() < deadline:
        random.shuffle(all_moduli)
        current_indices = []
        current_residues = []
        
        for i in range(n_mods):
            m = all_moduli[i]
            # Try to find a valid residue
            # We can try residues in a specific order. 
            # Random order of residues might help? 
            # Let's stick to 0..m-1 for determinism within a shuffle, 
            # but shuffle moduli gives diversity.
            
            valid_r = -1
            for r in range(m):
                if can_add_idx(r, i, current_indices, current_residues):
                    valid_r = r
                    break
            
            if valid_r != -1:
                current_indices.append(i)
                current_residues.append(valid_r)
                
        if len(current_indices) > best_len:
            best_len = len(current_indices)
            best_set = [(current_residues[k], all_moduli[current_indices[k]]) for k in range(len(current_indices))]
            
    # Convert to required format: list of [residue, modulus]
    result = [[r, m] for r, m in best_set]
    
    if not result:
        result = [[0, 2]]
        
    return result
