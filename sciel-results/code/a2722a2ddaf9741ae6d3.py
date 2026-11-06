import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def is_valid_assignment(residues, m_new, r_new):
        """Check if adding (m_new, r_new) to current residues is valid."""
        for m_prime, r_prime in residues.items():
            g = get_gcd(m_new, m_prime)
            if r_new % g == r_prime % g:
                return False
        return True

    def find_valid_residue(residues, m_new):
        """Find a residue for m_new that is valid against existing residues."""
        # Try all residues 0..m_new-1
        # Optimization: We can check constraints incrementally
        for r in range(m_new):
            if is_valid_assignment(residues, m_new, r):
                return r
        return -1

    def try_repair(residues, m_new, r_new):
        """
        Try to change residues of existing moduli to allow (m_new, r_new).
        Returns True if successful, updating residues.
        """
        # Identify conflicts: existing moduli m' such that r_new % gcd(m_new, m') == residues[m'] % gcd(m_new, m')
        conflicting_moduli = []
        for m_prime, r_prime in residues.items():
            g = get_gcd(m_new, m_prime)
            if r_new % g == r_prime % g:
                conflicting_moduli.append(m_prime)
        
        if not conflicting_moduli:
            return True # No conflicts, just add it (should have been caught earlier)

        # Try to resolve conflicts by changing residues of conflicting moduli
        # We can try random changes or systematic changes
        # For each conflicting m', try changing its residue to a different one
        # that doesn't conflict with m_new and doesn't create new conflicts with others
        
        # Simple heuristic: Pick a random conflicting modulus, try all its residues
        random.shuffle(conflicting_moduli)
        
        for m_conf in conflicting_moduli:
            g_conf = get_gcd(m_new, m_conf)
            forbidden_val = r_new % g_conf
            
            # Try all possible residues for m_conf
            for r_alt in range(m_conf):
                if r_alt % g_conf == forbidden_val:
                    continue # Still conflicts with m_new
                
                # Check if r_alt is valid against all OTHER moduli (except m_new, which we haven't added yet)
                # And valid against m_conf's other neighbors?
                # We need to ensure changing residues[m_conf] to r_alt doesn't break constraints with other existing moduli
                valid_alt = True
                for m_other, r_other in residues.items():
                    if m_other == m_conf:
                        continue
                    g_other = get_gcd(m_conf, m_other)
                    if r_alt % g_other == r_other % g_other:
                        valid_alt = False
                        break
                
                if valid_alt:
                    # Accept the change
                    residues[m_conf] = r_alt
                    # Note: We only changed one. There might be more conflicts.
                    # We return True to indicate "progress made", but we should re-check all conflicts
                    # Actually, if we change one, the others might still conflict.
                    # So we should loop until no conflicts or no changes possible.
                    # For now, let's say we made one step.
                    return True
            
        return False

    # Heuristic: Sort moduli by "difficulty"
    # Moduli with many divisors are harder?
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = {m: count_divisors(m) for m in moduli}
    
    # Strategy: 
    # 1. Greedy build with a specific order (e.g., primes first, then composites)
    # 2. Local search to add more
    
    for restart in range(500):
        if time.time() - start_time > timeout:
            break
        
        selected = []
        residues = {}
        
        # Order: Primes first (easiest), then others by descending divisor count?
        # Or random?
        # Let's try: Sort by value, but prioritize primes?
        # Actually, moduli with gcd 1 with many others are easier.
        # Primes are coprime to most things.
        
        available = moduli[:]
        
        # Strategy 0: Random
        # Strategy 1: Primes first
        # Strategy 2: Ascending value
        
        if restart % 3 == 0:
            random.shuffle(available)
        elif restart % 3 == 1:
            # Primes first
            primes = [m for m in available if all(m % i != 0 for i in range(2, int(math.isqrt(m)) + 1))]
            composites = [m for m in available if m not in primes]
            available = primes + composites
            random.shuffle(composites) # Randomize composites
        else:
            available.sort(key=lambda m: div_counts[m], reverse=True) # Hardest first?
        
        for m in available:
            if time.time() - start_time > timeout:
                break
            
            r = find_valid_residue(residues, m)
            if r != -1:
                residues[m] = r
                selected.append(m)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = [[residues[m2], m2] for m2 in selected]
        
        # Local Search: Try to add remaining moduli
        remaining = [m for m in moduli if m not in residues]
        random.shuffle(remaining)
        
        for m_new in remaining:
            if time.time() - start_time > timeout:
                break
            
            # Try to find a residue for m_new
            r_new = find_valid_residue(residues, m_new)
            if r_new != -1:
                residues[m_new] = r_new
                selected.append(m_new)
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_solution = [[residues[m2], m2] for m2 in selected]
            else:
                # Try to repair
                # Pick a random residue for m_new and try to fix conflicts
                # This is expensive, so do it sparingly
                if random.random() < 0.1: # Only 10% of the time
                    r_new = random.randint(0, m_new - 1)
                    # Try to repair
                    # We'll do a simple iterative repair
                    for _ in range(10):
                        if time.time() - start_time > timeout:
                            break
                        # Check if r_new is valid now
                        if is_valid_assignment(residues, m_new, r_new):
                            residues[m_new] = r_new
                            selected.append(m_new)
                            if len(selected) > best_count:
                                best_count = len(selected)
                                best_solution = [[residues[m2], m2] for m2 in selected]
                            break
                        else:
                            # Find a conflict
                            conflict_found = False
                            for m_prime, r_prime in residues.items():
                                g = get_gcd(m_new, m_prime)
                                if r_new % g == r_prime % g:
                                    # Try to change r_prime
                                    # Try all residues for m_prime
                                    changed = False
                                    for r_alt in range(m_prime):
                                        if r_alt % g == r_new % g:
                                            continue
                                        # Check if r_alt is valid for m_prime against others
                                        valid_alt = True
                                        for m_other, r_other in residues.items():
                                            if m_other == m_prime:
                                                continue
                                            g_other = get_gcd(m_prime, m_other)
                                            if r_alt % g_other == r_other % g_other:
                                                valid_alt = False
                                                break
                                        if valid_alt:
                                            residues[m_prime] = r_alt
                                            changed = True
                                            break
                                    if changed:
                                        conflict_found = True
                                        break
                            if not conflict_found:
                                # No conflict found? Should be valid.
                                # Or no change possible.
                                break

    if not best_solution:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
