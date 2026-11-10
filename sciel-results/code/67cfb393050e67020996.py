"""Find the minimal start s for 5 consecutive quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute roots: for each residue r in 1..1008, store one root x in 1..1008
    # such that x*x % p == r. Use smallest root for determinism.
    root_of = [0] * (p + 1)  # index 0..1008, root_of[r] = x or 0 if not a residue
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of[r] == 0:
            root_of[r] = x
    
    # For each s in 1..1004, check if s, s+1, s+2, s+3, s+4 are all quadratic residues
    # (i.e., root_of[s+i] != 0 for all i in 0..4)
    # We want the MINIMUM such s.
    
    # Efficient check: for each s, we need all of s, s+1, ..., s+4 to have non-zero roots.
    # Build a boolean array is_residue[r] = 1 if r is a nonzero QR mod p, else 0.
    is_residue = [0] * (p + 1)
    for r in range(1, p):
        if root_of[r] != 0:
            is_residue[r] = 1
    
    # Now scan s from 1 to 1004 and find the first s where all 5 consecutive are residues.
    # Note: s+4 must be <= 1008, so s <= 1004. This is already our range.
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if (is_residue[s] and is_residue[s+1] and 
            is_residue[s+2] and is_residue[s+3] and 
            is_residue[s+4]):
            roots = [root_of[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    # This should not happen, but handle it
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
