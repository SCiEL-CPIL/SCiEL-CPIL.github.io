import math
import time

def construct():
    deadline = time.time() + 85  # safety margin
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Start with all primes in [2, 128] - always valid with any residues
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # For primes, we can use residue 0 for all (since gcd of distinct primes is 1, condition is vacuous)
    # Actually let's use residue 1 for all primes - doesn't matter since gcd=1
    
    base_moduli = primes[:]
    base_residues = [1] * len(primes)
    
    # Candidate composites to try adding
    all_moduli = list(range(2, 129))
    candidates = [m for m in all_moduli if m not in base_moduli]
    
    best = list(zip(base_residues, base_moduli))
    best_count = len(base_moduli)
    
    # Try to add composites one by one
    # For a given set of (residue, modulus) pairs, check if we can add a new (r, m)
    # Condition: for all existing (r_i, m_i), r != r_i (mod gcd(m, m_i))
    
    def can_add(residues, moduli, new_r, new_m):
        for i, (r_i, m_i) in enumerate(zip(residues, moduli)):
            g = gcd(new_m, m_i)
            if new_r % g == r_i % g:
                return False
        return True
    
    # Greedy: try to add each composite, choosing a valid residue if exists
    current_residues = base_residues[:]
    current_moduli = base_moduli[:]
    
    # Sort candidates by "difficulty" - those with many prime factors first (more constraints)
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
    
    candidates.sort(key=num_distinct_prime_factors, reverse=True)
    
    for m in candidates:
        if time.time() > deadline - 5:
            break
        # Find a valid residue for m
        found = False
        for r in range(m):
            if can_add(current_residues, current_moduli, r, m):
                current_residues.append(r)
                current_moduli.append(m)
                found = True
                break
        if found:
            if len(current_moduli) > best_count:
                best_count = len(current_moduli)
                best = list(zip(current_residues[:], current_moduli[:]))
    
    # Try a second pass with different ordering to see if we can do better
    # Reverse order
    current_residues2 = base_residues[:]
    current_moduli2 = base_moduli[:]
    
    candidates_rev = candidates[:]
    candidates_rev.reverse()
    
    for m in candidates_rev:
        if time.time() > deadline - 5:
            break
        found = False
        for r in range(m):
            if can_add(current_residues2, current_moduli2, r, m):
                current_residues2.append(r)
                current_moduli2.append(m)
                found = True
                break
        if found:
            if len(current_moduli2) > best_count:
                best_count = len(current_moduli2)
                best = list(zip(current_residues2[:], current_moduli2[:]))
    
    # Try ascending order
    candidates_asc = sorted(candidates)
    current_residues3 = base_residues[:]
    current_moduli3 = base_moduli[:]
    
    for m in candidates_asc:
        if time.time() > deadline - 5:
            break
        found = False
        for r in range(m):
            if can_add(current_residues3, current_moduli3, r, m):
                current_residues3.append(r)
                current_moduli3.append(m)
                found = True
                break
        if found:
            if len(current_moduli3) > best_count:
                best_count = len(current_moduli3)
                best = list(zip(current_residues3[:], current_moduli3[:]))
    
    # Verify the best solution
    def verify(pairs):
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                r_i, m_i = pairs[i]
                r_j, m_j = pairs[j]
                g = gcd(m_i, m_j)
                if r_i % g == r_j % g:
                    return False
        # Check distinct moduli and canonical residues
        moduli_set = set()
        for r, m in pairs:
            if m in moduli_set:
                return False
            moduli_set.add(m)
            if not (0 <= r < m):
                return False
            if m < 2 or m > 128:
                return False
        return True
    
    if not verify(best):
        # Fallback to primes only
        best = [[1, p] for p in primes]
    
    return best

