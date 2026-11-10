"""Combined program: Fast randomized greedy + smart residue selection + local search for maximum disjoint residue classes."""
# EVOLVE-BLOCK-START
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

def _num_distinct_prime_factors(n):
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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute number of distinct prime factors
    pf_table = [0] * (MAX_MOD + 1)
    for i in range(2, MAX_MOD + 1):
        pf_table[i] = _num_distinct_prime_factors(i)
    
    best_count = 0
    best_result = None
    
    def try_add_smart(chosen, m, gcd_table, unused_moduli=None):
        """Find a valid residue for modulus m that minimizes future conflicts."""
        valid_residues = []
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                valid_residues.append(r)
        
        if not valid_residues:
            return None
        
        if len(valid_residues) == 1:
            return valid_residues[0]
        
        if unused_moduli is None:
            return valid_residues[0]
        
        # Score each valid residue by how many unused moduli it would block
        best_r = valid_residues[0]
        best_score = float('inf')
        
        for r in valid_residues:
            conflict_count = 0
            for m2 in unused_moduli:
                if m2 == m:
                    continue
                g = gcd_table[m][m2]
                if g > 1:
                    conflict_count += 1
            if conflict_count < best_score:
                best_score = conflict_count
                best_r = r
            if best_score == 0:
                break
        
        return best_r
    
    def greedy_construct(order, gcd_table):
        """Greedy: add moduli in given order, picking smart residue."""
        chosen = []
        used_set = set()
        for m in order:
            if m in used_set:
                continue
            unused = [mm for mm in range(2, MAX_MOD + 1) if mm not in used_set and mm != m]
            r = try_add_smart(chosen, m, gcd_table, unused)
            if r is not None:
                chosen.append((r, m))
                used_set.add(m)
        return chosen
    
    def local_search(chosen, gcd_table, deadline):
        """Try to improve by removing one and adding two."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            used_moduli = set(m for _, m in chosen)
            unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            for i in range(n):
                if time.time() > deadline:
                    break
                remaining = [c for j, c in enumerate(chosen) if j != i]
                remaining_used = set(m for _, m in remaining)
                unused_after = [m for m in unused if m not in remaining_used]
                
                added = 0
                new_chosen = list(remaining)
                for m in unused_after:
                    r = try_add_smart(new_chosen, m, gcd_table, unused_after)
                    if r is not None:
                        new_chosen.append((r, m))
                        added += 1
                    if added >= 2:
                        break
                
                if added >= 2:
                    chosen = new_chosen
                    improved = True
                    break
        
        return chosen
    
    # Multiple random restarts with diverse ordering strategies
    num_restarts = 500
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Diverse ordering strategies
        strategy = it % 5
        if strategy == 0:
            # Descending prime factors (harder first)
            order = sorted(moduli_list, key=lambda m: (-pf_table[m], random.random()))
        elif strategy == 1:
            # Ascending prime factors (easier first)
            order = sorted(moduli_list, key=lambda m: (pf_table[m], random.random()))
        elif strategy == 2:
            # Random
            order = moduli_list[:]
            random.shuffle(order)
        elif strategy == 3:
            # Descending modulus
            order = sorted(moduli_list, reverse=True)
        else:
            # Ascending modulus
            order = sorted(moduli_list)
        
        chosen = greedy_construct(order, gcd_table)
        chosen = local_search(chosen, gcd_table, deadline)
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Fallback: valid set of size 7
    fallback = [[0, 2], [1, 4], [3, 8], [5, 16], [7, 32], [15, 64], [23, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
    
    return best_result
# EVOLVE-BLOCK-END
