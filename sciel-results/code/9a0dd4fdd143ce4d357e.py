import math
import time
import random

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Pre-compute gcd table
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0
    
    # Step 1: Find maximum set of pairwise coprime moduli (backbone)
    # Two moduli conflict if gcd > 1. We want max independent set.
    # Use greedy: sort by number of conflicts (ascending), pick if compatible.
    
    def count_conflicts(m, chosen_set):
        count = 0
        for c in chosen_set:
            if math.gcd(m, c) > 1:
                count += 1
        return count
    
    # Greedy for pairwise coprime set
    def greedy_coprime_set():
        chosen = []
        remaining = sorted(all_mods)  # ascending
        for m in remaining:
            compatible = True
            for c in chosen:
                if math.gcd(m, c) > 1:
                    compatible = False
                    break
            if compatible:
                chosen.append(m)
        return chosen
    
    backbone = greedy_coprime_set()
    
    # For the backbone, any residues work since they're pairwise coprime.
    # Use residue 0 for simplicity.
    current_pairs = [(0, m) for m in backbone]
    
    # Step 2: Try to add each remaining modulus
    used_moduli = set(backbone)
    remaining_mods = [m for m in all_mods if m not in used_moduli]
    
    def try_add_modulus(m_new, pairs):
        """Try to find a residue for m_new compatible with all pairs."""
        for r in range(m_new):
            valid = True
            for (r2, m2) in pairs:
                if not fast_disjoint(r, m_new, r2, m2):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    # Greedily add remaining moduli
    for m_new in sorted(remaining_mods):
        r = try_add_modulus(m_new, current_pairs)
        if r is not None:
            current_pairs.append((r, m_new))
            used_moduli.add(m_new)
    
    best_pairs = current_pairs[:]
    best_len = len(best_pairs)
    
    # Step 3: Local search to improve
    def local_search(pairs, t_limit, seed_val):
        random.seed(seed_val)
        residues = [p[0] for p in pairs]
        moduli = [p[1] for p in pairs]
        best_count = len(residues)
        best_residues = residues[:]
        best_moduli = moduli[:]
        
        while time.time() < t_limit:
            improved = False
            
            # Try to add a new modulus
            used = set(moduli)
            available = [m for m in all_mods if m not in used]
            if available:
                random.shuffle(available)
                for m_new in available:
                    r = try_add_modulus(m_new, list(zip(residues, moduli)))
                    if r is not None:
                        residues.append(r)
                        moduli.append(m_new)
                        improved = True
                        break
            
            if not improved:
                # Try to reassign an existing residue
                if residues:
                    idx = random.randint(0, len(residues) - 1)
                    m = moduli[idx]
                    old_r = residues[idx]
                    for new_r in random.sample(range(m), min(m, 50)):
                        if new_r == old_r:
                            continue
                        valid = True
                        for j in range(len(residues)):
                            if j == idx:
                                continue
                            if not fast_disjoint(new_r, m, residues[j], moduli[j]):
                                valid = False
                                break
                        if valid:
                            residues[idx] = new_r
                            break
            
            # Try to swap: remove one, add two
            if len(residues) >= 2 and random.random() < 0.3:
                save_res = residues[:]
                save_mod = moduli[:]
                remove_idx = random.randint(0, len(residues) - 1)
                saved_r = residues.pop(remove_idx)
                saved_m = moduli.pop(remove_idx)
                
                used = set(moduli)
                available = [m for m in all_mods if m not in used]
                random.shuffle(available)
                added = 0
                for m_new in available[:10]:
                    r = try_add_modulus(m_new, list(zip(residues, moduli)))
                    if r is not None:
                        residues.append(r)
                        moduli.append(m_new)
                        added += 1
                        if added >= 2:
                            break
                
                # Try to re-add saved
                if saved_m not in moduli:
                    r = try_add_modulus(saved_m, list(zip(residues, moduli)))
                    if r is not None:
                        residues.append(r)
                        moduli.append(saved_m)
                        added += 1
                
                if len(residues) > best_count:
                    best_count = len(residues)
                    best_residues = residues[:]
                    best_moduli = moduli[:]
                else:
                    residues = save_res
                    moduli = save_mod
            
            cur_count = len(residues)
            if cur_count > best_count:
                best_count = cur_count
                best_residues = residues[:]
                best_moduli = moduli[:]
            
            if best_count == len(all_mods):
                break
        
        return [(best_residues[i], best_moduli[i]) for i in range(len(best_residues))]
    
    # Run local search with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        refined = local_search(best_pairs, deadline - 1.0, seed)
        if len(refined) > best_len:
            best_pairs = refined
            best_len = len(refined)
            if best_len == len(all_mods):
                break
    
    # Final verification
    result = []
    for (r, m) in best_pairs:
        if 0 <= r < m and 2 <= m <= MAX_MOD:
            result.append([r, m])
    
    if not result:
        result = [[0, 2]]
    
    return result
