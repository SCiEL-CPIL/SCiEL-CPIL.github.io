import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    moduli_range = list(range(20, 201))
    all_moduli = set(moduli_range)
    
    # Precompute GCDs
    gcd_table = {}
    for i in range(len(moduli_range)):
        for j in range(i + 1, len(moduli_range)):
            m1, m2 = moduli_range[i], moduli_range[j]
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        key = (min(m1, m2), max(m1, m2))
        return gcd_table.get(key, math.gcd(m1, m2))

    def find_valid_residue(m, residues_dict):
        """
        residues_dict: dict {m_i: r_i}
        Returns valid r for m or None
        """
        if not residues_dict:
            return 0
        
        # Optimization: check residues in a specific order?
        # Randomizing might help escape local minima in residue assignment
        # But for "fill", we just want ANY valid residue.
        # For "swap", we want to see if m_in can be added.
        
        candidates = list(range(m))
        random.shuffle(candidates)
        
        for r in candidates:
            ok = True
            for m_i, r_i in residues_dict.items():
                if m_i == m:
                    continue
                g = get_gcd(m, m_i)
                if (r - r_i) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    def calc_score(residues_dict):
        return sum(1.0 / m for m in residues_dict.keys())

    # 1. Greedy Initialization
    residues = {}
    # Sort by 1/m descending (m ascending)
    for m in sorted(all_moduli):
        r = find_valid_residue(m, residues)
        if r is not None:
            residues[m] = r
            
    current_score = calc_score(residues)
    best_score = current_score
    best_residues = dict(residues)
    
    missing = [m for m in all_moduli if m not in residues]
    included = list(residues.keys())
    
    # 2. Local Search
    no_improve_count = 0
    
    while time.time() < deadline:
        # Phase A: Try to fill all missing
        progress = True
        while progress and time.time() < deadline:
            progress = False
            # Try smallest missing first (highest value)
            for m in sorted(missing):
                if time.time() > deadline:
                    break
                r = find_valid_residue(m, residues)
                if r is not None:
                    residues[m] = r
                    missing.remove(m)
                    included.append(m)
                    progress = True
                    current_score += 1.0 / m
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = dict(residues)
        
        # If we filled everything, we are done (or near done)
        if not missing:
            break
            
        # Phase B: Kick/Swap
        # Try to remove a large included modulus to add a small missing one
        # We try multiple random kicks
        
        # Sort included by size (descending) to remove large ones
        # Sort missing by size (ascending) to add small ones
        
        # Heuristic: Try to swap out the largest included for the smallest missing
        # But we should randomize to explore.
        
        # Let's try a few specific swaps
        kicked = False
        # Try up to 50 random kicks per loop iteration
        for _ in range(50):
            if time.time() > deadline:
                break
            
            if not included or not missing:
                break
                
            # Choose m_out: Prefer larger moduli (lower value)
            # Randomly pick from the top 20% largest or just random
            m_out = random.choice(included)
            
            # Choose m_in: Prefer smaller moduli (higher value)
            # Randomly pick from the top 20% smallest or just random
            m_in = random.choice(missing)
            
            # If m_out is smaller than m_in, swapping is unlikely to improve score directly
            # but might unlock other additions.
            # Let's prioritize m_in < m_out for direct score gain.
            # If m_in >= m_out, we might still do it to explore, but let's bias towards m_in < m_out
            if m_in > m_out:
                # 50% chance to allow, else skip
                if random.random() > 0.5:
                    continue

            # Perform Swap
            old_r_out = residues.pop(m_out)
            
            r_in = find_valid_residue(m_in, residues)
            
            if r_in is not None:
                residues[m_in] = r_in
                missing.remove(m_in)
                included.remove(m_out)
                included.append(m_in)
                
                new_score = current_score - 1.0/m_out + 1.0/m_in
                
                if new_score > current_score:
                    current_score = new_score
                    kicked = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = dict(residues)
                    # Continue searching in this new state
                    break # Restart the fill phase
                else:
                    # Score decreased, revert
                    del residues[m_in]
                    residues[m_out] = old_r_out
                    missing.append(m_out)
                    included.remove(m_in)
                    included.append(m_out)
            else:
                # Couldn't add m_in, revert
                residues[m_out] = old_r_out
                # m_in was not added, so missing is unchanged
                # included is unchanged

        if not kicked:
            no_improve_count += 1
            if no_improve_count > 100:
                # Try a more aggressive restart or break
                # Let's try a random restart of the kick phase with different biases
                # Or just break if we've been stuck for too long
                if no_improve_count > 500:
                    break
        else:
            no_improve_count = 0

    # Final validation
    best_result = []
    for m, r in best_residues.items():
        best_result.append([m, r])
        
    # Verify
    for i in range(len(best_result)):
        for j in range(i + 1, len(best_result)):
            m1, r1 = best_result[i]
            m2, r2 = best_result[j]
            g = get_gcd(m1, m2)
            if (r1 - r2) % g == 0:
                # Invalid
                return [[20, 0]]
                
    return best_result
