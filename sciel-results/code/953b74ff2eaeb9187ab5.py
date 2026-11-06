import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_disjoint(r1, m1, r2, m2, gcd_table):
    """Check if residue class r1 mod m1 is disjoint from r2 mod m2."""
    g = gcd_table[m1][m2]
    return (r1 - r2) % g != 0

def _can_add(r, m, current, gcd_table):
    """Check if (r, m) is disjoint from all pairs in current."""
    for r2, m2 in current:
        g = gcd_table[m][m2]
        if (r - r2) % g == 0:
            return False
    return True

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin for 90s limit
    start_time = time.time()
    
    moduli = list(range(2, 129))
    n_mod = len(moduli)
    
    # Precompute gcd table indexed by modulus value (offset by 2)
    # gcd_table[m1][m2] = gcd(m1, m2)
    max_m = 128
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_table[m1][m2] = _gcd(m1, m2)
            else:
                gcd_table[m1][m2] = m1
    
    def greedy_init(perm):
        """Greedy: process moduli in given order, pick first valid residue for each."""
        selected = []
        for m in perm:
            # Try residues in a deterministic but varied order
            # Use a hash-based ordering to get diversity
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _can_add(r, m, selected, gcd_table):
                    selected.append((r, m))
                    break
        return selected
    
    def try_add_any(current, unused_mods):
        """Try to add any valid pair from unused moduli. Returns (new_solution, added) or (None, False)."""
        random.shuffle(unused_mods)
        for m in unused_mods:
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if _can_add(r, m, current, gcd_table):
                    return current + [(r, m)], True
        return current, False
    
    # Multiple random restarts with different orderings
    best_solution = None
    best_count = 0
    
    # Strategy: try several different greedy orderings
    orderings = []
    
    # Ordering 1: random
    perm1 = moduli[:]
    random.shuffle(perm1)
    orderings.append(perm1)
    
    # Ordering 2: by number of prime factors (fewer first - more constrained first)
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                count += 1
                temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count
    
    perm2 = sorted(moduli, key=lambda m: (num_prime_factors(m), m))
    orderings.append(perm2)
    
    # Ordering 3: by gcd with all others (more isolated first)
    def avg_gcd(m):
        total = 0
        for m2 in moduli:
            if m2 != m:
                total += gcd_table[m][m2]
        return total / (n_mod - 1)
    
    perm3 = sorted(moduli, key=lambda m: avg_gcd(m))
    orderings.append(perm3)
    
    # Ordering 4: reverse of perm2
    perm4 = list(reversed(perm2))
    orderings.append(perm4)
    
    # Ordering 5: another random
    perm5 = moduli[:]
    random.shuffle(perm5)
    orderings.append(perm5)
    
    for perm in orderings:
        if time.time() > deadline:
            break
        sol = greedy_init(perm)
        if len(sol) > best_count:
            best_count = len(sol)
            best_solution = sol[:]
    
    # Now do stochastic local search to improve
    if best_solution is not None:
        current = list(best_solution)
        used_mods = set(m for _, m in current)
        
        # Main SLS loop
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < deadline:
            # Phase 1: Try to add any valid pair
            unused_mods = [m for m in moduli if m not in used_mods]
            if unused_mods:
                new_sol, added = try_add_any(current, unused_mods)
                if added:
                    current = new_sol
                    used_mods = set(m for _, m in current)
                    if len(current) > best_count:
                        best_count = len(current)
                        best_solution = current[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    continue
            
            # Phase 2: No additions possible. Perform a kick.
            if len(current) <= 1:
                break
            
            # Kick: remove 1 or 2 random elements
            kick_size = 1 if random.random() < 0.6 else 2
            if len(current) > kick_size:
                indices_to_remove = random.sample(range(len(current)), kick_size)
                for idx in sorted(indices_to_remove, reverse=True):
                    _, m = current[idx]
                    used_mods.discard(m)
                    current.pop(idx)
            
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Reset: restart from best solution
                current = list(best_solution)
                used_mods = set(m for _, m in current)
                no_improve_count = 0
        
        # Final: try to add more to best_solution
        if best_solution is not None:
            final = list(best_solution)
            final_used = set(m for _, m in final)
            unused_final = [m for m in moduli if m not in final_used]
            random.shuffle(unused_final)
            for m in unused_final:
                if time.time() > deadline:
                    break
                residues = list(range(m))
                random.shuffle(residues)
                for r in residues:
                    if _can_add(r, m, final, gcd_table):
                        final.append((r, m))
                        break
            if len(final) > best_count:
                best_count = len(final)
                best_solution = final
    
    if best_solution is None:
        # Fallback: simple greedy
        best_solution = []
        for m in range(2, 129):
            for r in range(m):
                if _can_add(r, m, best_solution, gcd_table):
                    best_solution.append((r, m))
                    break
    
    return [[r, m] for r, m in best_solution]
