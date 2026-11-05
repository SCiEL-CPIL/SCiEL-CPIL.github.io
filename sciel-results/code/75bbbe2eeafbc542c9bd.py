import math

def construct():
    max_mod = 128
    
    # Get all primes up to max_mod
    primes = []
    for i in range(2, max_mod + 1):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Precompute prime factors for each modulus
    prime_factors = {}
    for m in range(2, max_mod + 1):
        factors = set()
        n = m
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                factors.add(p)
                while n % p == 0:
                    n //= p
        if n > 1:
            factors.add(n)
        prime_factors[m] = factors
        
    # Sort moduli by smallest prime factor descending
    # This prioritizes numbers with large prime factors (less constrained)
    moduli = list(range(2, max_mod + 1))
    
    def get_spf(m):
        factors = prime_factors[m]
        return min(factors) if factors else m
        
    moduli_sorted = sorted(moduli, key=lambda m: -get_spf(m))
    
    selected = []
    counts = {p: 0 for p in primes}
    
    for m in moduli_sorted:
        can_add = True
        for p in prime_factors[m]:
            if counts[p] >= p:
                can_add = False
                break
        if can_add:
            selected.append(m)
            for p in prime_factors[m]:
                counts[p] += 1
                
    if not selected:
        return [[0, 2]]
        
    # Now assign residues
    # For each prime p, assign distinct residues to the selected multiples of p
    # We need r_m mod p to be distinct for all m in selected divisible by p.
    
    # We can do this prime by prime.
    # For each prime p, let M_p be the list of selected moduli divisible by p.
    # We assign residues 0, 1, ..., len(M_p)-1 to them arbitrarily?
    # But r_m is a single number. It must satisfy r_m ≡ v_{m,p} (mod p) for ALL p dividing m.
    # This is a system of congruences.
    # Since p's are distinct, we can use CRT.
    
    # However, we need to be careful. The value r_m must be consistent.
    # Let's collect constraints for each m.
    # constraints[m] = list of (value, modulus)
    
    # We need to assign values v_{m,p} for each p|m.
    # To make it simple, we can just assign v_{m,p} = index of m in M_p.
    # But we need to ensure consistency? No, CRT guarantees a solution exists.
    # We just need to compute it.
    
    # Let's group selected moduli by prime factors.
    # For each prime p, get list of m in selected where p|m.
    # Assign distinct residues mod p to these m's.
    
    residue_constraints = {m: [] for m in selected}
    
    for p in primes:
        multiples = [m for m in selected if m % p == 0]
        # Assign distinct residues 0, 1, ..., len(multiples)-1
        # We can shuffle to avoid bias? Or just sequential.
        # Sequential is fine.
        for i, m in enumerate(multiples):
            residue_constraints[m].append((i, p))
            
    # Now for each m, solve CRT for its constraints
    result = []
    for m in selected:
        if not residue_constraints[m]:
            # Should not happen for m >= 2
            r = 0
        else:
            # Solve x ≡ a_i (mod m_i)
            # m_i are primes, so pairwise coprime
            # x = sum(a_i * M_i * inv(M_i, m_i)) mod M
            # M = product of all primes dividing m
            M = 1
            for val, mod in residue_constraints[m]:
                M *= mod
                
            x = 0
            for val, mod in residue_constraints[m]:
                M_i = M // mod
                # Inverse of M_i mod mod
                # Since mod is prime, use pow(M_i, -1, mod)
                inv = pow(M_i, -1, mod)
                x += val * M_i * inv
                
            x %= M
            # x is a solution modulo M.
            # We need r in [0, m-1].
            # Since M divides m? No, M is the product of distinct prime factors of m.
            # m is a multiple of M.
            # So x is a valid residue modulo M.
            # We can just take r = x % m?
            # Wait, if we take r = x, it satisfies x ≡ val (mod p) for all p|m.
            # Is r < m?
            # M <= m. So x < M <= m.
            # So r = x is in [0, m-1].
            r = x
            
        result.append([r, m])
        
    return result
