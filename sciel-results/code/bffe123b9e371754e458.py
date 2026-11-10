import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Sort by number of prime factors descending, then by value
    sorted_moduli = sorted(moduli, key=lambda x: (-count_prime_factors(x), x))
    
    best_solution = []
    best_size = 0
    
    # Precompute gcds for all pairs to speed up
    # n = 127
    # gcd_cache[i][j] for moduli[i], moduli[j]
    n_mods = len(sorted_moduli)
    gcd_cache = [[0]*n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i+1, n_mods):
            g = math.gcd(sorted_moduli[i], sorted_moduli[j])
            gcd_cache[i][j] = g
            gcd_cache[j][i] = g

    def is_compatible(r_new, m_new_idx, existing_indices, existing_residues):
        # existing_indices: list of indices in sorted_moduli
        # existing_residues: list of residues
        for k_idx, k_res in zip(existing_indices, existing_residues):
            g = gcd_cache[m_new_idx][k_idx]
            if r_new % g == k_res % g:
                return False
        return True

    def backtrack(idx, current_indices, current_residues):
        nonlocal best_solution, best_size
        if time.time() > deadline:
            return
        
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_solution = [[current_residues[i], sorted_moduli[current_indices[i]]] for i in range(len(current_indices))]
        
        if idx >= n_mods:
            return
            
        # Option 1: Skip
        backtrack(idx + 1, current_indices, current_residues)
        
        # Option 2: Include
        m = sorted_moduli[idx]
        # Try all residues
        # Optimization: If we find one valid residue, we proceed.
        # Do we need to try all? 
        # If we include m, the choice of r affects future.
        # To be thorough, we should try all valid residues.
        # But this increases branching factor by m.
        # Given the time limit, let's try just the first valid residue.
        # If this is too greedy, we can add a "try all" flag if time permits.
        
        for r in range(m):
            if is_compatible(r, idx, current_indices, current_residues):
                backtrack(idx + 1, current_indices + [idx], current_residues + [r])
                break # Only try the first valid residue to keep depth manageable
        
    backtrack(0, [], [])
    
    if not best_solution:
        best_solution = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
# EVOLVE-BLOCK-END
