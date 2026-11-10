"""Rewritten program: Structured search for maximum disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    best_count = 0
    best_result = []
    
    def greedy_solve(order, use_random_residues=False):
        """Given an ordering of moduli, greedily assign residues."""
        selected = []
        for m in order:
            found = False
            if use_random_residues:
                residues = list(range(m))
                random.shuffle(residues)
            else:
                residues = list(range(m))
            
            for r in residues:
                valid = True
                for r_s, m_s in selected:
                    g = gcd_table[m][m_s]
                    if (r - r_s) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append((r, m))
                    found = True
                    break
        return selected
    
    # Phase 1: Randomized greedy with full moduli set
    for _ in range(300):
        if time.time() > deadline:
            break
        order = moduli_list[:]
        random.shuffle(order)
        result = greedy_solve(order, use_random_residues=True)
        if len(result) > best_count:
            best_count = len(result)
            best_result = [[r, m] for r, m in result]
    
    # Phase 2: Focus on even moduli (coprime-free set)
    even_mods = [m for m in moduli_list if m % 2 == 0]
    for _ in range(300):
        if time.time() > deadline:
            break
        order = even_mods[:]
        random.shuffle(order)
        result = greedy_solve(order, use_random_residues=True)
        if len(result) > best_count:
            best_count = len(result)
            best_result = [[r, m] for r, m in result]
    
    # Phase 3: Try to add odd moduli to the best even-only solution
    if best_result and time.time() < deadline:
        selected_pairs = list(best_result)
        selected_mods = set(m for r, m in selected_pairs)
        
        # Sort odd moduli by how many selected moduli they share factors with
        odd_mods = [m for m in moduli_list if m % 2 != 0 and m not in selected_mods]
        
        def shared_factor_count(m):
            count = 0
            for r_s, m_s in selected_pairs:
                if gcd_table[m][m_s] > 1:
                    count += 1
            return count
        
        odd_mods.sort(key=shared_factor_count, reverse=True)
        
        for m in odd_mods:
            if time.time() > deadline:
                break
            
            # Check if m shares a factor > 1 with ALL selected moduli
            compatible_with_all = True
            for r_s, m_s in selected_pairs:
                if gcd_table[m][m_s] == 1:
                    compatible_with_all = False
                    break
            
            if not compatible_with_all:
                continue
            
            # Try to find a valid residue
            for r in range(m):
                valid = True
                for r_s, m_s in selected_pairs:
                    g = gcd_table[m][m_s]
                    if (r - r_s) % g == 0:
                        valid = False
                        break
                if valid:
                    new_selected = selected_pairs + [(r, m)]
                    if len(new_selected) > best_count:
                        best_count = len(new_selected)
                        best_result = [[r_s, m_s] for r_s, m_s in new_selected]
                        selected_pairs = new_selected
                        selected_mods.add(m)
                    break
    
    # Phase 4: Local search - try swapping residues to enable more moduli
    if best_result and time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Take current best and try random re-optimization
            selected_mods = [m for r, m in best_result]
            trial_mods = selected_mods[:]
            random.shuffle(trial_mods)
            
            trial_selected = []
            for m in trial_mods:
                found = False
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    valid = True
                    for r_s, m_s in trial_selected:
                        g = gcd_table[m][m_s]
                        if (r - r_s) % g == 0:
                            valid = False
                            break
                    if valid:
                        trial_selected.append((r, m))
                        found = True
                        break
                if not found:
                    break
            
            if len(trial_selected) > best_count:
                best_count = len(trial_selected)
                best_result = [[r, m] for r, m in trial_selected]
    
    # Fallback
    if not best_result:
        best_result = [[0, 2], [1, 4], [3, 8], [5, 16], [7, 32], [15, 64], [23, 128]]
    
    return best_result
# EVOLVE-BLOCK-END
