import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli)

    # Precompute gcds
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Precompute for each pair (i,j) with gcd>1, the set of residues r_i that conflict with each r_j
    # For pair (i,j) with g = gcd(m_i, m_j), residue r_i conflicts with r_j iff (r_i - r_j) % g == 0
    # i.e., r_i ≡ r_j (mod g)
    # So for fixed j and r_j, the conflicting r_i values are those where r_i % g == r_j % g
    
    # For each index i, for each other index j, precompute: for each residue r_j of j,
    # which residues of i are forbidden?
    # This is expensive to precompute fully. Instead, we'll use forward checking during backtracking.

    # Variable ordering heuristic: process moduli in order of increasing number of "conflicting" other moduli
    # A modulus m_i conflicts with m_j if gcd(m_i, m_j) > 1
    # Count how many other moduli share a factor > 1 with each modulus
    conflict_count = [0] * n_mods
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            if gcds[i][j] > 1:
                conflict_count[i] += 1
                conflict_count[j] += 1

    # Sort indices by conflict count (most constrained first)
    order = sorted(range(n_mods), key=lambda i: -conflict_count[i])

    # Backtracking with constraint propagation
    residues = [-1] * n_mods  # -1 means unassigned
    best_size = 0
    best_solution = []

    def is_time_up():
        return time.time() > deadline - 2.0

    def backtrack(pos, current_size):
        nonlocal best_size, best_solution
        
        if is_time_up():
            return False
        
        if current_size > best_size:
            best_size = current_size
            sol = []
            for idx in order:
                if residues[idx] >= 0:
                    sol.append([residues[idx], moduli[idx]])
            best_solution = sol
        
        if pos >= n_mods:
            return True  # Found complete solution
        
        idx = order[pos]
        m = moduli[idx]
        
        # Try residues in a smart order: prefer residues that don't conflict with already assigned
        # We'll try all residues 0..m-1
        found = False
        for r in range(m):
            # Check if r is compatible with all already assigned residues
            compatible = True
            for prev_pos in range(pos):
                prev_idx = order[prev_pos]
                if residues[prev_idx] < 0:
                    continue
                g = gcds[idx][prev_idx]
                if g > 1 and (r - residues[prev_idx]) % g == 0:
                    compatible = False
                    break
            
            if compatible:
                residues[idx] = r
                if backtrack(pos + 1, current_size + 1):
                    return True
                residues[idx] = -1
        
        # If no residue works for this index, try skipping this index
        # But wait - we want to maximize, so we should try to include it if possible
        # If we can't include it, we skip it (leave as -1) and continue
        return False

    # Run backtracking
    backtrack(0, 0)

    if best_solution:
        return best_solution
    else:
        return [[0, 2]]
