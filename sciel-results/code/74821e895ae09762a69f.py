import time
import math
import random

def _is_disjoint(r1, m1, r2, m2, gcd_table=None):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    if gcd_table is not None:
        g = gcd_table[m1][m2]
    else:
        g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _csp_backtrack(available_moduli, gcd_table, deadline, max_depth=50):
    """
    CSP-style backtracking with forward checking.
    Tries to build a solution by selecting moduli and residues with MRV heuristic.
    """
    if time.time() > deadline:
        return []
    
    # State: chosen list of (residue, modulus)
    # We'll use a recursive approach with forward checking
    
    def backtrack(chosen, remaining):
        if time.time() > deadline:
            return chosen
        if not remaining:
            return chosen
        
        # MRV: pick modulus with fewest valid residues
        best_idx = -1
        best_count = float('inf')
        best_valid = None
        
        for i, m in enumerate(remaining):
            valid = []
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    g = gcd_table[m][m2]
                    if (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    valid.append(r)
            if len(valid) < best_count:
                best_count = len(valid)
                best_idx = i
                best_valid = valid
                if best_count == 0:
                    break
        
        if best_count == 0:
            return chosen  # Dead end, return what we have
        
        # Try each valid residue for the chosen modulus
        m = remaining[best_idx]
        new_remaining = remaining[:best_idx] + remaining[best_idx+1:]
        
        # Shuffle for diversity
        shuffled_valid = best_valid[:]
        random.shuffle(shuffled_valid)
        
        for r in shuffled_valid:
            new_chosen = chosen + [(r, m)]
            result = backtrack(new_chosen, new_remaining)
            if len(result) > len(chosen):
                return result
        
        return chosen
    
    return backtrack([], list(available_moduli))

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin well under 90s kill
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute number of distinct prime factors for each modulus
    def num_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    pf_table = [0] * (MAX_MOD + 1)
    for i in range(2, MAX_MOD + 1):
        pf_table[i] = num_distinct_prime_factors(i)
    
    best_count = 0
    best_result = None
    
    def get_valid_residues(chosen, m, gcd_table):
        """Get all valid residues for modulus m given current assignments."""
        valid = []
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                valid.append(r)
        return valid
    
    def greedy_construct(order, gcd_table, deadline):
        """Greedy: add moduli in given order, picking residue that blocks least."""
        chosen = []
        used_set = set()
        for m in order:
            if time.time() > deadline:
                break
            valid = get_valid_residues(chosen, m, gcd_table)
            if not valid:
                continue
            # Pick residue that blocks fewest future options
            best_r = valid[0]
            best_score = float('inf')
            for r in valid:
                score = 0
                for m2 in range(2, MAX_MOD + 1):
                    if m2 in used_set or m2 == m:
                        continue
                    g = gcd_table[m][m2]
                    if g > 1:
                        # Count residues of m2 blocked by (r, m)
                        blocked = (m2 + g - 1 - r % g) // g  # ceil((m2 - r%g)/g) approximation
                        score += blocked
                if score < best_score:
                    best_score = score
                    best_r = r
            chosen.append((best_r, m))
            used_set.add(m)
        return chosen
    
    def local_search(chosen, gcd_table, deadline):
        """Try to improve by removing one and adding two, or swapping residues."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            used_moduli = set(m for _, m in chosen)
            unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            # Strategy 1: Remove one, try to add two
            for i in range(n):
                if time.time() > deadline:
                    break
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                unused_after = [m for m in unused if m not in remaining_used]
                
                added = 0
                new_chosen = list(remaining)
                for m in unused_after:
                    valid = get_valid_residues(new_chosen, m, gcd_table)
                    if valid:
                        new_chosen.append((valid[0], m))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    chosen = new_chosen
                    improved = True
                    break
            
            if improved:
                continue
            
            # Strategy 2: Try changing residue of one element to enable adding another
            for i in range(n):
                if time.time() > deadline:
                    break
                r_old, m_old = chosen[i]
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                
                # Try different residues for m_old
                for r_new in range(m_old):
                    if r_new == r_old:
                        continue
                    # Check if (r_new, m_old) is valid with remaining
                    ok = True
                    for (r2, m2) in remaining:
                        g = gcd_table[m_old][m2]
                        if (r_new - r2) % g == 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_chosen = remaining + [(r_new, m_old)]
                    new_used = set(m for _, m in new_chosen)
                    
                    # Try to add an unused modulus
                    for m_add in unused:
                        if m_add in new_used:
                            continue
                        valid = get_valid_residues(new_chosen, m_add, gcd_table)
                        if valid:
                            new_chosen.append((valid[0], m_add))
                            if len(new_chosen) > len(chosen):
                                chosen = new_chosen
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            # Strategy 3: Replace one modulus with a different unused one
            for i in range(n):
                if time.time() > deadline:
                    break
                old_r, old_m = chosen[i]
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                
                for m_new in unused:
                    if m_new in remaining_used:
                        continue
                    valid = get_valid_residues(remaining, m_new, gcd_table)
                    if valid:
                        new_chosen = remaining + [(valid[0], m_new)]
                        new_used = set(m for _, m in new_chosen)
                        
                        # Try to add more
                        still_unused = [m for m in range(2, MAX_MOD + 1) if m not in new_used]
                        added_more = 0
                        for m_add in still_unused:
                            v = get_valid_residues(new_chosen, m_add, gcd_table)
                            if v:
                                new_chosen.append((v[0], m_add))
                                added_more += 1
                            if added_more >= 1:
                                break
                        
                        if len(new_chosen) > len(chosen):
                            chosen = new_chosen
                            improved = True
                            break
                if improved:
                    break
        
        return chosen
    
    def swap_local_search(chosen, gcd_table, deadline):
        """Try more aggressive local improvements: 2-for-1 swaps and residue optimization."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            used_moduli = set(m for _, m in chosen)
            unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            # Try removing two and adding three
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    remaining = [c for k, c in enumerate(chosen) if k != i and k != j]
                    remaining_used = set(m for _, m in remaining)
                    unused_after = [m for m in unused if m not in remaining_used]
                    
                    added = 0
                    new_chosen = list(remaining)
                    for m in unused_after:
                        valid = get_valid_residues(new_chosen, m, gcd_table)
                        if valid:
                            new_chosen.append((valid[0], m))
                            added += 1
                        if added >= 3:
                            break
                    if added >= 3:
                        chosen = new_chosen
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try optimizing residues: for each element, try all valid residues
            # and see if it enables adding more
            for i in range(n):
                if time.time() > deadline:
                    break
                r_old, m_old = chosen[i]
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                
                best_r = r_old
                best_added = 0
                
                for r_try in range(m_old):
                    # Check validity
                    ok = True
                    for (r2, m2) in remaining:
                        g = gcd_table[m_old][m2]
                        if (r_try - r2) % g == 0:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_chosen = remaining + [(r_try, m_old)]
                    new_used = set(m for _, m in new_chosen)
                    
                    # Count how many unused moduli can be added
                    added = 0
                    for m_add in unused:
                        if m_add in new_used:
                            continue
                        valid = get_valid_residues(new_chosen, m_add, gcd_table)
                        if valid:
                            added += 1
                    
                    if added > best_added:
                        best_added = added
                        best_r = r_try
                
                if best_r != r_old:
                    chosen[i] = (best_r, m_old)
                    # Try to add the enabled moduli
                    new_used = set(m for _, m in chosen)
                    for m_add in unused:
                        if m_add in new_used:
                            continue
                        valid = get_valid_residues(chosen, m_add, gcd_table)
                        if valid:
                            chosen.append((valid[0], m_add))
                            improved = True
                            break
        
        return chosen
    
    # Multiple random restarts with different strategies
    num_restarts = 300
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with various strategies
        if it % 7 == 0:
            # Sort by descending number of prime factors
            order = sorted(moduli_list, key=lambda m: (-pf_table[m], random.random()))
        elif it % 7 == 1:
            # Sort by ascending number of prime factors
            order = sorted(moduli_list, key=lambda m: (pf_table[m], random.random()))
        elif it % 7 == 2:
            # Random
            order = moduli_list[:]
            random.shuffle(order)
        elif it % 7 == 3:
            # Sort by descending modulus
            order = sorted(moduli_list, reverse=True)
        elif it % 7 == 4:
            # Sort by ascending modulus
            order = sorted(moduli_list)
        elif it % 7 == 5:
            # Sort by descending pf, then random
            order = sorted(moduli_list, key=lambda m: (-pf_table[m], m))
            random.shuffle(order)
        else:
            # Sort by ascending pf, then random
            order = sorted(moduli_list, key=lambda m: (pf_table[m], m))
            random.shuffle(order)
        
        chosen = greedy_construct(order, gcd_table, deadline)
        chosen = local_search(chosen, gcd_table, deadline)
        chosen = swap_local_search(chosen, gcd_table, deadline)
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Final CSP backtracking attempt with remaining time
    if time.time() < deadline - 5:
        used_moduli = set(m for _, m in (best_result if best_result else []))
        available = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
        if available:
            csp_result = _csp_backtrack(available, gcd_table, deadline - 2)
            if csp_result:
                # Merge with existing best
                existing = best_result if best_result else []
                existing_moduli = set(m for _, m in existing)
                merged = list(existing)
                for r, m in csp_result:
                    if m not in existing_moduli:
                        merged.append([r, m])
                        existing_moduli.add(m)
                if len(merged) > best_count:
                    best_count = len(merged)
                    best_result = merged
    
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
