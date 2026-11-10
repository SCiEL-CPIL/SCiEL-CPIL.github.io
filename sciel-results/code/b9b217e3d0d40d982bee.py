"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Strategy: Use multiples of 6 to ensure gcd >= 6.
    # This allows for more classes than powers of 2 (which are limited by gcd 2 constraints).
    moduli = [m for m in range(6, 129, 6)]
    
    # Assign residues. 
    # For any pair i, j, gcd(m_i, m_j) >= 6.
    # We need r_i != r_j mod gcd(m_i, m_j).
    # A simple heuristic: r_m = m % 6? No, m is divisible by 6, so r_m=0. All same. Bad.
    # Try r_m = (m // 6) % 6?
    # Let's try a greedy assignment.
    
    residues = {}
    for m in moduli:
        found = False
        for r in range(m):
            valid = True
            for m2, r2 in residues.items():
                import math
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    valid = False
                    break
            if valid:
                residues[m] = r
                found = True
                break
        if not found:
            # If no residue found, skip this modulus (or handle error)
            # But we want to maximize, so we should pick moduli that allow assignment.
            # Greedy might fail. 
            pass
            
    # This greedy might be slow or fail. 
    # Let's stick to the known good 7 if this doesn't work quickly.
    # But let's try a specific pattern for multiples of 6.
    # r_m = (m // 6) % 6
    # Check: m=6, r=0. m=12, r=1. gcd=6. 0!=1 mod 6. OK.
    # m=6, r=0. m=18, r=2. gcd=6. 0!=2 mod 6. OK.
    # m=12, r=1. m=18, r=2. gcd=6. 1!=2 mod 6. OK.
    # m=12, r=1. m=24, r=3. gcd=12. 1%12=1, 3%12=3. 1!=3. OK.
    
    pairs = []
    for m in moduli:
        r = (m // 6) % 6
        # Verify
        valid = True
        for [r2, m2] in pairs:
            import math
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                valid = False
                break
        if valid:
            pairs.append([r, m])
            
    if len(pairs) > 7:
        return pairs
    else:
        # Fallback to powers of 2
        moduli = [1 << k for k in range(1, 8)]
        residues = [1 << (k-1) for k in range(1, 8)]
        return [[r, m] for r, m in zip(residues, moduli)]
# EVOLVE-BLOCK-END
