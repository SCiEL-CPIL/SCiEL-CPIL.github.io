import math

MAX_MOD = 128

def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    # Step 1: Generate candidates and their prime factors
    candidates = []
    for m in range(2, MAX_MOD + 1):
        pf = _prime_factors(m)
        candidates.append((m, pf))
    
    # Sort by number of distinct prime factors, then by value
    # This prioritizes "cheap" moduli
    candidates.sort(key=lambda x: (len(x[1]), x[0]))
    
    # Step 2: Greedy selection
    # C_p tracks how many selected moduli are divisible by prime p
    # Constraint: C_p <= p
    C = {}
    selected = []
    
    for m, pf in candidates:
        # Check if we can add m
        can_add = True
        for p in pf:
            current_count = C.get(p, 0)
            if current_count >= p:
                can_add = False
                break
        
        if can_add:
            selected.append((m, pf))
            for p in pf:
                C[p] = C.get(p, 0) + 1
                
    # Step 3: Assign residues
    # For each prime p, assign distinct residues to moduli in S_p
    # S_p = {m in selected : p in pf(m)}
    
    # First, determine v[m][p] for all m, p|m
    # v[m][p] is the residue of m modulo p
    # We need v[m][p] to be distinct for all m in S_p
    
    # Group selected moduli by prime factor
    S = {}
    for m, pf in selected:
        for p in pf:
            if p not in S:
                S[p] = []
            S[p].append(m)
    
    # Assign values
    # val[m][p] = assigned residue mod p
    val = {}
    
    for p, mods in S.items():
        # Sort mods to make assignment deterministic
        mods.sort()
        for i, m in enumerate(mods):
            if m not in val:
                val[m] = {}
            val[m][p] = i % p  # i is distinct for each m in S_p, and i < len(S_p) <= p
    
    # Now for each m, solve CRT for r_m
    result = []
    for m, pf in selected:
        # Constraints: r_m ≡ val[m][p] (mod p) for all p in pf
        # L = product of pf
        L = 1
        for p in pf:
            L *= p
            
        # Solve r ≡ a_i (mod m_i)
        # Using simple CRT since moduli are pairwise coprime primes
        r = 0
        M_total = 1
        for p in pf:
            a = val[m][p]
            # r = r + M_total * t * (a - r) * inv(M_total, p)
            # Or just brute force since L is small?
            # L is product of primes dividing m. Max m=128. L <= 128.
            # Brute force 0..L-1 is very fast.
            pass
        
        # Brute force is safer and fast enough for L <= 128
        r_found = None
        for r in range(L):
            ok = True
            for p in pf:
                if r % p != val[m][p]:
                    ok = False
                    break
            if ok:
                r_found = r
                break
        
        if r_found is None:
            # Should not happen
            r_found = 0
            
        # r_found is in [0, L). Since L divides m, r_found < m.
        result.append([r_found, m])
        
    return result
