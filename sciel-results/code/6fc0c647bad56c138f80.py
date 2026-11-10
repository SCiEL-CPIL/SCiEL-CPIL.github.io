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

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
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
    
    def try_add_smart(chosen, m, gcd_table, unused_moduli=None):
        """Try to find a valid residue for modulus m that minimizes future conflicts."""
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
        
        # Score each valid residue by how many future moduli it would block
        # Use a heuristic: count conflicts with unused moduli
        if unused_moduli is None:
            return valid_residues[0]  # Fallback to first
        
        best_r = valid_residues[0]
        best_score = float('inf')
        
        for r in valid_residues:
            # Count how many unused moduli would have at least one valid residue
            # with this choice (lower is better - we want to leave options open)
            conflict_count = 0
            for m2 in unused_moduli:
                if m2 == m:
                    continue
                g = gcd_table[m][m2]
                # Check if r blocks all residues for m2
                # Actually, we just count if r creates a conflict pattern
                # Simple heuristic: count gcd values > 1
                if g > 1:
                    conflict_count += 1
            # Prefer residues that minimize the number of moduli with gcd > 1
            # This is a rough proxy for "leaving more options"
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
            unused = [mm for mm in range(2, MAX_MOD + 1) if mm not in used_set and mm != m]
            r = try_add_smart(chosen, m, gcd_table, unused)
            if r is not None:
                chosen.append((r, m))
                used_set.add(m)
        return chosen
    
    def local_search(chosen, gcd_table, deadline):
        """Try to improve by removing one and adding two, or swapping."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            used_moduli = set(m for _, m in chosen)
            unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
            
            # Try removing one element and adding two new ones
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
    
    # Multiple random restarts with different orderings
    num_restarts = 300
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering with various strategies
        if it % 5 == 0:
            # Sort by descending number of prime factors (harder to place first)
            order = sorted(moduli_list, key=lambda m: (-pf_table[m], random.random()))
        elif it % 5 == 1:
            # Sort by ascending number of prime factors (easier to place first)
            order = sorted(moduli_list, key=lambda m: (pf_table[m], random.random()))
        elif it % 5 == 2:
            # Random
            order = moduli_list[:]
            random.shuffle(order)
        elif it % 5 == 3:
            # Sort by descending modulus
            order = sorted(moduli_list, reverse=True)
        else:
            # Sort by ascending modulus
            order = sorted(moduli_list)
        
        chosen = greedy_construct(order, gcd_table)
        chosen = local_search(chosen, gcd_table, deadline)
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
