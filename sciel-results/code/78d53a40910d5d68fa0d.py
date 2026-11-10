import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute GCDs for moduli 20..200
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    best_score = 0.0
    best_solution = []
    
    # Run multiple randomized greedy trials
    # Each trial tries to include all moduli in increasing order
    # If a modulus cannot be included with any residue, it is skipped.
    # Randomizing the choice of residue helps avoid "bad" choices that block many future moduli.
    
    while time.time() < deadline:
        selected = []
        # Store indices into moduli list for selected items to speed up gcd lookup
        selected_indices = []
        selected_residues = []
        current_score = 0.0
        
        for i, m in enumerate(moduli):
            valid_rs = []
            # Check all possible residues for m
            for r in range(m):
                is_valid = True
                # Check against all previously selected
                for k in range(len(selected_indices)):
                    idx_prev = selected_indices[k]
                    r_prev = selected_residues[k]
                    g = gcd_matrix[i][idx_prev]
                    if (r % g) == (r_prev % g):
                        is_valid = False
                        break
                if is_valid:
                    valid_rs.append(r)
                    
            if valid_rs:
                # Pick a random valid residue to diversify search
                r = random.choice(valid_rs)
                selected.append([m, r])
                selected_indices.append(i)
                selected_residues.append(r)
                current_score += 1.0 / m
            # else: skip this modulus
            
        if current_score > best_score:
            best_score = current_score
            best_solution = selected
            # If we found a solution with all moduli, we can stop early?
            # Max possible score is sum(1/m for m in 20..200)
            if len(selected) == n_mods:
                # It's theoretically impossible to do better than including all moduli
                # So this is the global optimum for this set of moduli
                break

    return best_solution
# EVOLVE-BLOCK-END
