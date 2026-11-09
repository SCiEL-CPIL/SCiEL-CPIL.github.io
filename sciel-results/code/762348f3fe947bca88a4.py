"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import gcd

def construct():
    MAX_M = 128
    # Moduli range from 2 to 128
    moduli = list(range(2, MAX_M + 1))
    num_moduli = len(moduli)
    # Map modulus value to index 0..num_moduli-1
    m_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute forbidden masks
    # forbidden_masks[i][j][r] is a bitmask for modulus moduli[j]
    # indicating which residues are forbidden if we pick residue r for modulus moduli[i]
    # i, j in 0..num_moduli-1
    # r in 0..moduli[i]-1
    
    # We only need this for i != j. If i == j, picking any r forbids all others, 
    # but we handle that by removing the modulus from available.
    
    forbidden_masks = [[None] * num_moduli for _ in range(num_moduli)]
    
    for i in range(num_moduli):
        m_i = moduli[i]
        for j in range(num_moduli):
            if i == j:
                continue
            m_j = moduli[j]
            g = gcd(m_i, m_j)
            masks_for_m_i = [0] * m_i
            # For each residue r in m_i, compute forbidden residues in m_j
            for r in range(m_i):
                r_mod_g = r % g
                mask = 0
                # Residues k in m_j such that k % g == r_mod_g
                # k starts at r_mod_g, then r_mod_g + g, etc.
                k = r_mod_g
                while k < m_j:
                    mask |= (1 << k)
                    k += g
                masks_for_m_i[r] = mask
            forbidden_masks[i][j] = masks_for_m_i
            
    # Initial allowed masks
    # allowed[i] is a bitmask of allowed residues for moduli[i]
    # Initially all residues 0..m-1 are allowed
    initial_allowed = [0] * num_moduli
    for i in range(num_moduli):
        m = moduli[i]
        initial_allowed[i] = (1 << m) - 1
        
    best_size = 0
    best_solution = []
    
    # available_mask: bitmask of moduli indices that are still active
    # Initially all active
    all_active_mask = (1 << num_moduli) - 1
    
    # To help MRV, we might maintain counts, but calculating popcount is fast enough
    # Python int.bit_count() is available in 3.8+? Yes, or bin(x).count('1')
    
    def search(available_mask, allowed_masks, current_size, current_solution):
        nonlocal best_size, best_solution
        
        if current_size > best_size:
            best_size = current_size
            best_solution = current_solution.copy()
            
        # Pruning bound
        # Upper bound on additional moduli we can add
        # Simple bound: number of available moduli
        remaining_count = available_mask.bit_count()
        if current_size + remaining_count <= best_size:
            return
            
        if remaining_count == 0:
            return
            
        # MRV Heuristic: Find modulus with fewest allowed residues
        min_count = float('inf')
        best_idx = -1
        # Iterate through available moduli
        # We can iterate through bits of available_mask
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            
            count = allowed_masks[idx].bit_count()
            if count == 0:
                # This should not happen if we maintain available_mask correctly,
                # but if it does, we should have removed it.
                continue
            if count < min_count:
                min_count = count
                best_idx = idx
                if count == 1:
                    break # Can't do better than 1
                    
        if best_idx == -1:
            return
            
        m_val = moduli[best_idx]
        allowed_res = allowed_masks[best_idx]
        
        # Iterate through allowed residues for this modulus
        res_mask = allowed_res
        while res_mask:
            lsb = res_mask & -res_mask
            r = lsb.bit_length() - 1
            res_mask ^= lsb
            
            # Try picking (r, m_val)
            new_solution = current_solution + [[r, m_val]]
            
            # Update allowed masks for other moduli
            new_allowed = allowed_masks[:] # Copy
            new_available = available_mask & ~(1 << best_idx) # Remove best_idx
            
            # Update masks for all other available moduli
            # We only need to update those that are still available
            m_iter = new_available
            while m_iter:
                lsb_idx = m_iter & -m_iter
                idx_j = lsb_idx.bit_length() - 1
                m_iter ^= lsb_idx
                
                # Apply forbidden mask
                # forbidden_masks[best_idx][idx_j][r]
                forbid = forbidden_masks[best_idx][idx_j][r]
                new_allowed[idx_j] &= ~forbid
                
                # If no residues left, remove from available
                if new_allowed[idx_j] == 0:
                    new_available &= ~lsb_idx
                    
            search(new_available, new_allowed, current_size + 1, new_solution)
            
    search(all_active_mask, initial_allowed, 0, [])
    
    # Return the best solution found
    return best_solution

# EVOLVE-BLOCK-END
