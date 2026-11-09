"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # gcd[m1_idx][m2_idx]
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_disjoint_set(classes):
        """Check if all pairs are disjoint."""
        for i in range(len(classes)):
            m1, r1 = classes[i]
            for j in range(i + 1, len(classes)):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True

    def can_add(classes, m, r):
        """Check if (m, r) can be added to classes without conflict."""
        for m2, r2 in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def score(classes):
        """Calculate exact score as Fraction."""
        if not classes:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for m, r in classes:
            s += Fraction(1, m)
        return s

    def greedy_construct():
        """
        Greedy construction:
        1. Sort moduli by 1/m (descending).
        2. For each modulus, try to find a residue r that doesn't conflict.
           To maximize future flexibility, we might prefer residues that are 'hard' to place?
           Actually, any valid residue works for the score. 
           However, the choice of residue affects which future classes can be added.
           We can try a few residues for each modulus to see which leads to a better set.
           But for speed, let's just pick the first valid one, or try all and pick the one 
           that allows the most future additions? That's expensive.
           
           Alternative: Randomized Greedy with Restarts.
        """
        best_set = []
        best_score = Fraction(0, 1)
        
        # Sort moduli by value (ascending m -> descending 1/m)
        sorted_moduli = sorted(moduli)
        
        for _ in range(50): # 50 restarts
            # Shuffle residues to break symmetry
            import random
            random.seed()
            
            current_set = []
            # We process moduli in order of decreasing weight (1/m)
            for m in sorted_moduli:
                # Try random residues
                residues = list(range(m))
                random.shuffle(residues)
                
                added = False
                for r in residues:
                    if can_add(current_set, m, r):
                        current_set.append([m, r])
                        added = True
                        break
                
                if not added:
                    # If we can't add this modulus, skip it
                    continue
            
            s = score(current_set)
            if s > best_score:
                best_score = s
                best_set = [x[:] for x in current_set]
                
            if time.time() > deadline:
                break
                
        return best_set

    def local_search(initial_set):
        """
        Improve the set by swapping or adding.
        1. Try adding any missing modulus with a valid residue.
        2. Try swapping an existing modulus with a missing one if it improves score.
        """
        current_set = [x[:] for x in initial_set]
        current_score = score(current_set)
        
        # Indices of moduli in current set
        current_mods = {m for m, r in current_set}
        
        # List of candidate moduli not in set
        candidates = [m for m in moduli if m not in current_mods]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try adding a candidate
            # Sort candidates by 1/m descending
            candidates_sorted = sorted(candidates, key=lambda x: 1/x, reverse=True)
            
            for m_new in candidates_sorted:
                # Try to find a valid residue for m_new
                # To maximize chance, try random residues
                found = False
                # Limit attempts for speed
                for _ in range(10):
                    import random
                    r_new = random.randint(0, m_new - 1)
                    if can_add(current_set, m_new, r_new):
                        # Adding always improves score since 1/m > 0
                        current_set.append([m_new, r_new])
                        current_score += Fraction(1, m_new)
                        current_mods.add(m_new)
                        candidates.remove(m_new)
                        improved = True
                        found = True
                        break
                
                if found:
                    # Re-evaluate candidates list? No, just continue loop
                    # But we should restart the candidate loop to check others
                    # Actually, adding one might block others, so restart is safer
                    break
            
            if improved:
                continue
                
            # 2. Try swapping
            # For each existing class, try replacing it with a candidate
            # This is O(N * M * Residues). Might be slow.
            # Only do this if we have time.
            if time.time() < deadline - 10:
                for i, (m_old, r_old) in enumerate(current_set):
                    # Try candidates with higher weight? Or just any candidate?
                    # Since we are maximizing sum, we only swap if 1/m_new > 1/m_old
                    # i.e., m_new < m_old
                    swap_candidates = [m for m in candidates if m < m_old]
                    if not swap_candidates:
                        continue
                    
                    for m_new in swap_candidates:
                        # Try to find residue for m_new
                        for _ in range(5):
                            import random
                            r_new = random.randint(0, m_new - 1)
                            # Check if m_new can replace m_old
                            # Remove m_old, check if m_new fits
                            temp_set = current_set[:i] + current_set[i+1:]
                            if can_add(temp_set, m_new, r_new):
                                # Perform swap
                                current_set[i] = [m_new, r_new]
                                current_score += Fraction(1, m_new) - Fraction(1, m_old)
                                current_mods.remove(m_old)
                                current_mods.add(m_new)
                                candidates.remove(m_new)
                                candidates.append(m_old)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
        return current_set

    # Initial greedy
    best_set = greedy_construct()
    
    # Local search refinement
    if time.time() < deadline:
        refined_set = local_search(best_set)
        if score(refined_set) > score(best_set):
            best_set = refined_set
            
    # One more pass of greedy + local search with different random seeds?
    # The greedy_construct already did 50 restarts.
    # Let's try a few more local searches from different greedy starts if time permits.
    
    if time.time() < deadline - 5:
        # Try 10 more greedy starts
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            # Re-use logic from greedy_construct but single pass
            import random
            random.seed()
            current_set = []
            sorted_moduli = sorted(moduli)
            for m in sorted_moduli:
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if can_add(current_set, m, r):
                        current_set.append([m, r])
                        break
            
            refined = local_search(current_set)
            s_refined = score(refined)
            s_best = score(best_set)
            if s_refined > s_best:
                best_set = refined

    return best_set
# EVOLVE-BLOCK-END
