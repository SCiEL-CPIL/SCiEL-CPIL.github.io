"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Pre-compute gcd table
    max_mod = 128
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            gcd_table[m1][m2] = gcd(m1, m2)
    
    all_moduli = list(range(2, max_mod+1))
    
    def _check_pair(r1, m1, r2, m2):
        g = gcd_table[m1][m2]
        return (r1 - r2) % g != 0
    
    def _is_valid_set(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i+1, n):
                r1, m1 = classes[i]
                r2, m2 = classes[j]
                if not _check_pair(r1, m1, r2, m2):
                    return False
        return True
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multi-start randomized greedy with backtracking hints
    random.seed(42)
    max_iters = 50000
    iter_count = 0
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        iter_count += 1
        
        # Shuffle moduli
        moduli = all_moduli[:]
        random.shuffle(moduli)
        
        current_set = []
        
        for m in moduli:
            # Check if m can be added
            compatible = True
            for r_old, m_old in current_set:
                if gcd_table[m][m_old] == 1:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Find a residue r for m that works
            found_r = -1
            # Try residues in a smart order: prefer residues that are "far" from existing ones
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for r_old, m_old in current_set:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                current_set.append([found_r, m])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = [list(c) for c in current_set]
    
    # Strategy 2: Local search / improvement on the best set found
    if best_size > 0:
        # Try to add more classes to the best set
        used_moduli = set(m for _, m in best_set)
        for m in all_moduli:
            if m in used_moduli:
                continue
            if time.time() > deadline:
                break
            
            # Check if m is compatible with all current classes
            compatible = True
            for r_old, m_old in best_set:
                if gcd_table[m][m_old] == 1:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            # Find a residue
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in best_set:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                best_set.append([found_r, m])
                best_size += 1
    
    # Strategy 3: Try specific structured approaches
    # Approach A: Use only moduli that are multiples of a small set of primes
    # This increases the chance of non-coprime pairs (which is what we want for more classes)
    
    # Let's try a deterministic greedy that prioritizes moduli with many shared factors
    # Moduli with more prime factors are more likely to be non-coprime with others
    
    def _prime_factor_count(n):
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
    
    # Sort moduli by number of prime factors (descending) - these are "denser"
    moduli_by_density = sorted(all_moduli, key=lambda m: -_prime_factor_count(m))
    
    # Try this ordering
    current_set2 = []
    for m in moduli_by_density:
        if time.time() > deadline:
            break
        
        compatible = True
        for r_old, m_old in current_set2:
            if gcd_table[m][m_old] == 1:
                compatible = False
                break
        
        if not compatible:
            continue
        
        found_r = -1
        for r in range(m):
            ok = True
            for r_old, m_old in current_set2:
                g = gcd_table[m][m_old]
                if (r - r_old) % g == 0:
                    ok = False
                    break
            if ok:
                found_r = r
                break
        
        if found_r != -1:
            current_set2.append([found_r, m])
    
    if len(current_set2) > best_size:
        best_size = len(current_set2)
        best_set = [list(c) for c in current_set2]
    
    # Approach B: Focus on moduli that share a common small prime factor
    # E.g., all even moduli, or all multiples of 3
    
    for common_prime in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        
        # Get moduli divisible by common_prime
        candidate_moduli = [m for m in all_moduli if m % common_prime == 0]
        random.shuffle(candidate_moduli)
        
        current_set3 = []
        for m in candidate_moduli:
            compatible = True
            for r_old, m_old in current_set3:
                if gcd_table[m][m_old] == 1:
                    compatible = False
                    break
            
            if not compatible:
                continue
            
            found_r = -1
            for r in range(m):
                ok = True
                for r_old, m_old in current_set3:
                    g = gcd_table[m][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                current_set3.append([found_r, m])
        
        if len(current_set3) > best_size:
            best_size = len(current_set3)
            best_set = [list(c) for c in current_set3]
    
    # Strategy 4: Backtracking search for small extensions
    # Try to extend the best set by removing one class and adding two
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        if len(best_set) == 0:
            break
        
        # Try removing each class and see if we can add more
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            
            trial_set = [c for j, c in enumerate(best_set) if j != i]
            removed_mod = best_set[i][1]
            removed_res = best_set[i][0]
            
            # Try to add back removed_mod with a different residue, or add a new mod
            # First, try to find a better residue for removed_mod
            found_r = -1
            for r in range(removed_mod):
                if r == removed_res:
                    continue
                ok = True
                for r_old, m_old in trial_set:
                    g = gcd_table[removed_mod][m_old]
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    found_r = r
                    break
            
            if found_r != -1:
                new_set = trial_set + [[found_r, removed_mod]]
                # Try to add more classes
                used_moduli = set(m for _, m in new_set)
                for m in all_moduli:
                    if m in used_moduli:
                        continue
                    compatible = True
                    for r_old, m_old in new_set:
                        if gcd_table[m][m_old] == 1:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    found_r2 = -1
                    for r in range(m):
                        ok = True
                        for r_old, m_old in new_set:
                            g = gcd_table[m][m_old]
                            if (r - r_old) % g == 0:
                                ok = False
                                break
                        if ok:
                            found_r2 = r
                            break
                    if found_r2 != -1:
                        new_set.append([found_r2, m])
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = [list(c) for c in new_set]
    
    return best_set
# EVOLVE-BLOCK-END
