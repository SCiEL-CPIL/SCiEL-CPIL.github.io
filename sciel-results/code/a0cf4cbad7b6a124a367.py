"""Rewritten program to find the minimum start s for 5 consecutive quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Compute quadratic residues and one root for each
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        # We only care about non-zero residues
        if r != 0:
            # Store the smallest root for efficiency/canonicity, though any valid root works
            if r not in roots:
                roots[r] = x
            
    # Scan for the minimum s
    # s must be such that s, s+1, s+2, s+3, s+4 are all in [1, p-1] and are QRs
    # Max s is p-5 = 1004
    for s in range(1, p - 4):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # They must be non-zero, which is guaranteed if they are in roots (keys are non-zero)
        # Note: s+i could be p? No, s <= p-5, so s+4 <= p-1.
        # Also s >= 1, so s+i >= 1.
        
        # Optimization: Check membership in the dictionary
        # If any is not in roots, it's not a QR (or is 0, which isn't in roots)
        if s in roots and (s+1) in roots and (s+2) in roots and (s+3) in roots and (s+4) in roots:
            return {
                'start': s,
                'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]
            }
            
    # Fallback: If no run of 5 is found (should not happen for p=1009), 
    # the seed program logic (descending) might have been intended to find *a* run?
    # But the task says "Minimize s". 
    # If no minimum is found, we might return the best found or raise error.
    # Given p=1009, runs of 5 are expected.
    raise RuntimeError("No run of 5 consecutive quadratic residues found")
# EVOLVE-BLOCK-END
