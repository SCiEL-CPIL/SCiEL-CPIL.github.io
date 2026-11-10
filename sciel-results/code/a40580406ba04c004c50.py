import random
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Candidate moduli: Even numbers in [20, 200]
    # This ensures gcd(m_i, m_j) >= 2 for all pairs, making disjointness possible.
    moduli = [m for m in range(20, 201) if m % 2 == 0]
    
    best_score = 0.0
    best_set = []
    
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Run multiple random restarts
    # We use randomization to escape local minima in the greedy assignment
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        selected_residues = []
        selected_indices = []
        
        for i, m in enumerate(moduli):
            # Find valid residues
            # Optimization: Iterate r and check constraints
            # If m is small, this is fast.
            valid_found = False
            # To diversify, we can shuffle the range or pick randomly
            # But checking all is needed to know if it's valid.
            # We can collect valid ones and pick one.
            valid_residues = []
            
            for r in range(m):
                is_valid = True
                for idx in selected_indices:
                    g = gcds[i][idx]
                    r2 = selected_residues[selected_indices.index(idx)] # This is slow!
                    # Better: keep a list of (m, r)
                    
                # Let's restructure to keep (m, r) pairs
            # Restructure loop
            pass

    # Corrected Loop Structure
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        selected = [] # List of [m, r]
        # Map modulus to index for gcd lookup
        # moduli is sorted, so index is implicit if we iterate in order
        
        for i, m in enumerate(moduli):
            valid_residues = []
            for r in range(m):
                is_valid = True
                for j in range(len(selected)):
                    m2, r2 = selected[j]
                    # Need gcd(m, m2). We have precomputed gcds[i][idx_of_m2]
                    # But we don't have idx_of_m2 easily in the selected list.
                    # Let's store indices in selected.
                    pass

    # Let's use indices in selected
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        selected_indices = []
        selected_residues = []
        
        for i, m in enumerate(moduli):
            valid_residues = []
            for r in range(m):
                is_valid = True
                for k in range(len(selected_indices)):
                    idx = selected_indices[k]
                    r2 = selected_residues[k]
                    g = gcds[i][idx]
                    if (r - r2) % g == 0:
                        is_valid = False
                        break
                if is_valid:
                    valid_residues.append(r)
                    
            if valid_residues:
                # Pick one
                r = random.choice(valid_residues)
                selected_indices.append(i)
                selected_residues.append(r)
                
        # Calculate score
        score = 0.0
        for idx in selected_indices:
            score += 1.0 / moduli[idx]
            
        if score > best_score:
            best_score = score
            best_set = [[moduli[idx], selected_residues[k]] for k, idx in enumerate(selected_indices)]
            
    return best_set
