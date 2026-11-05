import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    # Precompute gcds
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        mi = moduli[i]
        for j in range(i + 1, n_mods):
            mj = moduli[j]
            g = math.gcd(mi, mj)
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    best_count = 0
    best_solution = []

    def try_greedy(order):
        selected = []
        # selected holds (modulus_idx, residue)
        
        for m_idx in order:
            m_val = moduli[m_idx]
            found = False
            # Try residues in random order
            residues = list(range(m_val))
            random.shuffle(residues)
            
            for r in residues:
                # Check compatibility with all selected
                ok = True
                for s_idx, s_r in selected:
                    g = gcd_matrix[m_idx][s_idx]
                    if (r - s_r) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append((m_idx, r))
                    found = True
                    break
            
            # If not found, we just skip this modulus in this iteration.
            # This is a greedy maximal set construction.
            
        return selected

    attempts = 0
    # Pre-compute a heuristic order: sort by descending "connectivity" (sum of gcds)
    # This might help, but random restarts with random orders are also good.
    # Let's mix: some random, some heuristic.
    
    # Calculate sum of gcds for each modulus
    gcd_sums = [0] * n_mods
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j:
                gcd_sums[i] += gcd_matrix[i][j]
    
    # Sort indices by gcd_sum descending
    heuristic_order = sorted(range(n_mods), key=lambda x: gcd_sums[x], reverse=True)
    
    while time.time() < deadline:
        attempts += 1
        if attempts % 2 == 0:
            order = list(heuristic_order)
            random.shuffle(order) # Perturb heuristic
        else:
            order = list(range(n_mods))
            random.shuffle(order)
            
        sol = try_greedy(order)
        
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = sol
            # If we find a very large set, we can increase deadline or stop?
            # Keep searching to ensure we hit the max.

    result = []
    for m_idx, r in best_solution:
        m_val = moduli[m_idx]
        result.append([r, m_val])
        
    if not result:
        result = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
        
    return result
