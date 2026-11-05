import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    max_modulus = 128
    moduli = list(range(2, max_modulus + 1))
    n_moduli = len(moduli)
    
    # Precompute gcds
    gcds = {}
    for i in range(n_moduli):
        m1 = moduli[i]
        for j in range(i + 1, n_moduli):
            m2 = moduli[j]
            gcds[(m1, m2)] = math.gcd(m1, m2)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcds.get((m1, m2), math.gcd(m1, m2))

    # Precompute conflict masks
    # conflict_mask[i][j] is a bitmask of residues r_j that conflict with ANY residue of m_i? No.
    # We need: given r_i, which r_j are forbidden?
    # forbidden_mask[i][j][r_i] = bitmask of r_j in [0, m_j) such that r_i == r_j (mod gcd(m_i, m_j))
    
    # To save memory and time, we can compute this on the fly or precompute per pair.
    # There are ~8000 pairs. For each pair, m_j is up to 128.
    # Precomputing a 128x128x128 array might be heavy but doable? 
    # 127 * 127 * 128 bytes if using bytes? Or just compute on the fly.
    # On-the-fly: For a given r_i, m_i, m_j, we can compute the forbidden residues for m_j.
    # r_j is forbidden if r_j % g == r_i % g, where g = gcd(m_i, m_j).
    # The number of such residues is m_j // g.
    # We can create a mask quickly.
    
    def get_forbidden_mask(r_i, m_i, m_j):
        g = get_gcd(m_i, m_j)
        rem = r_i % g
        mask = 0
        for r_j in range(m_j):
            if r_j % g == rem:
                mask |= (1 << r_j)
        return mask

    # Ordering: Sort moduli to improve pruning.
    # Heuristic: Moduli with many small factors (high gcd with many others) are more restrictive.
    # Let's try sorting by decreasing number of divisors.
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    div_counts = {m: num_divisors(m) for m in moduli}
    ordered_moduli = sorted(moduli, key=lambda m: -div_counts[m])
    
    # Map original index to new index
    mod_index = {m: i for i, m in enumerate(ordered_moduli)}
    
    # available[i] is the bitmask of available residues for ordered_moduli[i]
    available = [0] * n_moduli
    for i, m in enumerate(ordered_moduli):
        available[i] = (1 << m) - 1
        
    best_solution = []
    best_len = 0
    current_selection = [None] * n_moduli # (r, m) or None
    
    def backtrack(idx):
        nonlocal best_solution, best_len
        
        if time.time() > deadline:
            return
        
        # Upper bound: if we can't beat best, stop
        # Simple bound: number of remaining moduli that still have available residues
        remaining_count = 0
        for j in range(idx, n_moduli):
            if available[j] != 0:
                remaining_count += 1
        if len(best_solution) + remaining_count <= best_len:
            return
            
        if idx == n_moduli:
            if len(best_solution) > best_len:
                best_len = len(best_solution)
                best_solution = list(current_selection)
            return
        
        m = ordered_moduli[idx]
        
        # Try each available residue
        # Iterate through set bits in available[idx]
        mask = available[idx]
        while mask:
            if time.time() > deadline:
                return
            # Get lowest set bit
            lsb = mask & -mask
            r = lsb.bit_length() - 1
            mask ^= lsb
            
            # Check if this residue is compatible with all previously selected
            # Since we update 'available' array, if r is in available[idx], it is compatible
            # with all previous selections.
            
            # Save state
            old_available = [0] * n_moduli
            for j in range(idx + 1, n_moduli):
                old_available[j] = available[j]
                
            # Update available for future moduli
            valid = True
            for j in range(idx + 1, n_moduli):
                m_j = ordered_moduli[j]
                if available[j] == 0:
                    continue # Already dead, no need to update
                forbidden = get_forbidden_mask(r, m, m_j)
                new_avail = available[j] & ~forbidden
                if new_avail == 0:
                    # Dead end
                    valid = False
                    break
                available[j] = new_avail
                
            if valid:
                current_selection[idx] = (r, m)
                backtrack(idx + 1)
                current_selection[idx] = None
                
                # Restore state
                for j in range(idx + 1, n_moduli):
                    available[j] = old_available[j]

    # Run backtracking
    backtrack(0)
    
    if best_solution:
        # Filter out Nones and format
        result = [[r, m] for r, m in best_solution if m is not None]
        return result
    else:
        # Fallback if backtracking failed to find anything (should not happen)
        return [[0, 2]]
