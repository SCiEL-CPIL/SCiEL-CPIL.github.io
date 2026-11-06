def construct():
    p = 1009
    # Map each non-zero quadratic residue to a valid root
    # We store the smallest positive root for determinism, though any valid root works
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the first root found (smallest x)
            if r not in roots:
                roots[r] = x
    
    # Set of valid residues for O(1) lookup
    valid_residues = set(roots.keys())
    
    # Search for the minimum s
    # s must be such that s, s+1, ..., s+4 are all in valid_residues
    # Constraints: 1 <= s <= 1004 (so s+4 <= 1008, no wrap-around to 0 or beyond p-1 needed for the value check, 
    # but we must ensure s+i != 0 mod p. Since s>=1 and s<=1004, s+i is in [1, 1008], so never 0 mod 1009)
    
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # Using a list comprehension or all() for clarity
        if all((s + i) in valid_residues for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
            
    # Fallback (should not happen for p=1009)
    raise RuntimeError("No run found")
