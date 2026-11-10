"""Rewritten program to find the minimal start s for a run of 5 quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute a mapping from non-zero quadratic residues to a valid root in [1, p-1]
    # We iterate x from 1 to p-1. For each x, r = x*x % p.
    # If r is non-zero, we store it. We want to ensure we have a root for every QR.
    # Since p is prime, every non-zero QR has exactly two roots.
    # We just need one.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Store the first root found (smallest x) for determinism, though any is valid
            if r not in roots:
                roots[r] = x
    
    # Search for the minimal s in [1, 1004]
    # We need s, s+1, s+2, s+3, s+4 to be non-zero quadratic residues.
    # Max s is 1004 because s+4 <= 1008 (since residues are in 1..1008, 0 is excluded)
    for s in range(1, 1005):
        # Check if s+i is a valid key in roots for i in 0..4
        # Note: s+i must be <= 1008. Since s <= 1004, s+4 <= 1008.
        # Also s+i >= 1.
        if all((s + i) in roots for i in range(5)):
            return {
                'start': s,
                'roots': [roots[s + i] for i in range(5)]
            }
            
    # This should never happen for p=1009, but raise an error just in case
    raise RuntimeError("No valid run found")
