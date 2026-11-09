"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    
    def is_compatible(r_new, m_new, chosen):
        """Check if (r_new, m_new) is compatible with all in chosen."""
        for r, m in chosen:
            g = gcd(m_new, m)
            if r_new % g == r % g:
                return False
        return True
    
    def find_compatible_residue(m_new, chosen):
        """Find a residue for m_new compatible with chosen, or None."""
        for r in range(m_new):
            if is_compatible(r, m_new, chosen):
                return r
        return None
    
    # Phase 1: Multiple greedy constructions with different orderings
    best_result = []
    best_count = 0
    
    random.seed(42)
    
    # Deterministic orderings
    orderings = []
    orderings.append(sorted(all_moduli))
    orderings.append(sorted(all_moduli, reverse=True))
    # Sort by number of divisors (fewer divisors = less restrictive)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    orderings.append(sorted(all_moduli, key=num_divisors))
    orderings.append(sorted(all_moduli, key=lambda m: -m.bit_count()))
    
    # Random orderings
    for _ in range(200):
        order = all_moduli[:]
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        chosen = []
        for m in order:
            if time.time() > deadline - 20:
                break
            # Try residues in random order for diversity
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_compatible(r, m, chosen):
                    chosen.append((r, m))
                    break
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_result = chosen[:]
    
    # Phase 2: Local search - try to add more moduli
    if time.time() < deadline - 10:
        chosen_mod_set = set(m for r, m in best_result)
        remaining = [m for m in all_moduli if m not in chosen_mod_set]
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            random.shuffle(remaining)
            for m in remaining[:]:
                if time.time() > deadline - 10:
                    break
                r = find_compatible_residue(m, best_result)
                if r is not None:
                    best_result.append((r, m))
                    remaining.remove(m)
                    best_count += 1
                    improved = True
    
    # Phase 3: Local search - try swapping residues to free up space for more moduli
    if time.time() < deadline - 5:
        chosen_mod_set = set(m for r, m in best_result)
        remaining = [m for m in all_moduli if m not in chosen_mod_set]
        
        # Try swapping residues of existing classes to make room for new ones
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            if not remaining:
                break
            
            # Pick a random modulus from remaining
            m_new = random.choice(remaining)
            
            # Try to find a swap that allows adding m_new
            found_swap = False
            for i in range(len(best_result)):
                r_i, m_i = best_result[i]
                # Try changing residue of class i
                for r_new_i in range(m_i):
                    if r_new_i == r_i:
                        continue
                    # Check if new residue for class i is still compatible with others
                    compatible = True
                    for j in range(len(best_result)):
                        if j == i:
                            continue
                        r_j, m_j = best_result[j]
                        g = gcd(m_i, m_j)
                        if r_new_i % g == r_j % g:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Now check if we can add m_new with some residue
                    temp_result = best_result[:]
                    temp_result[i] = (r_new_i, m_i)
                    r_m_new = find_compatible_residue(m_new, temp_result)
                    if r_m_new is not None:
                        # Accept the swap and add new class
                        best_result[i] = (r_new_i, m_i)
                        best_result.append((r_m_new, m_new))
                        remaining.remove(m_new)
                        best_count += 1
                        found_swap = True
                        break
                if found_swap:
                    break
    
    if best_count > 0:
        return [[r, m] for r, m in best_result]
    
    return [[1, 2]]
# EVOLVE-BLOCK-END
