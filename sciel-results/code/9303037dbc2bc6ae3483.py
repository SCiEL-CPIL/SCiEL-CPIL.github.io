import math

def construct():
    max_mod = 128
    # We'll try to build a large set of moduli with pairwise disjoint residue classes
    # Key: two classes (r1,m1) and (r2,m2) are disjoint iff gcd(m1,m2) does not divide (r2-r1)
    # If gcd(m1,m2)==1, they always intersect, so we can't have both.
    
    # Strategy: select a subset of moduli where pairwise gcd > 1,
    # then assign residues via backtracking/greedy.
    
    # First, find a large clique in the "gcd>1" graph
    # Use greedy: sort moduli by number of gcd>1 connections, pick greedily
    
    moduli = list(range(2, max_mod + 1))
    
    # Build adjacency: which moduli can coexist (gcd > 1)
    # For greedy selection, use a heuristic order
    # Prefer moduli with many small prime factors (more flexible)
    
    def prime_factors(n):
        pf = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf.add(d)
                n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf
    
    # For each modulus, compute its prime factors
    pf_map = {m: prime_factors(m) for m in moduli}
    
    # Greedy: try to include moduli in order of "richness" (more prime factors, smaller primes)
    # Sort by: (-len(pf), min(pf) if pf else 999, m)
    sorted_moduli = sorted(moduli, key=lambda m: (-len(pf_map[m]), min(pf_map[m]) if pf_map[m] else 999, m))
    
    selected = []  # list of (residue, modulus)
    selected_moduli = set()
    
    # Try each modulus in order, see if we can assign a residue
    for m in sorted_moduli:
        # Check if m can coexist with all already selected (gcd > 1)
        can_coexist = True
        for (r_s, m_s) in selected:
            if math.gcd(m, m_s) == 1:
                can_coexist = False
                break
        if not can_coexist:
            continue
        
        # Try to find a residue r in [0, m) such that for all (r_s, m_s) in selected,
        # gcd(m, m_s) does NOT divide (r - r_s)
        found = False
        for r in range(m):
            valid = True
            for (r_s, m_s) in selected:
                g = math.gcd(m, m_s)
                if (r - r_s) % g == 0:
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                selected_moduli.add(m)
                found = True
                break
        
        if not found:
            # Try other residues? We already tried all. Skip this modulus.
            pass
    
    # If we didn't get enough, try a different order or backtracking
    # For now, return what we have
    if not selected:
        return [[1, 2]]
    
    return selected
