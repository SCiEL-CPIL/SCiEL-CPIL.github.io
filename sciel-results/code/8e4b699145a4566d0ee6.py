"""Rewritten program: bitmask-based search for minimal run-start of 5 consecutive quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute: for each residue r in 1..p-1, store a root x such that x^2 % p == r
    # We need residues to be in 1..1008 (exclude 0), so we only care about non-zero residues
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Build a bitmask: bit i is set if residue i is a quadratic residue (and non-zero)
    # We index residues 1..1008, so bit position (r-1) corresponds to residue r
    # Actually, let's use a boolean array for clarity and speed
    is_residue = [False] * p  # index 0..1008
    for r in roots:
        is_residue[r] = True
    
    # We need to find the smallest s in 1..1004 such that
    # s, s+1, s+2, s+3, s+4 are all quadratic residues (non-zero)
    # i.e., is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]
    
    # For speed, use a bitmask approach:
    # Create an integer bitmask where bit i (0-indexed) is 1 if residue (i+1) is a QR
    # Then for start s, we need bits s-1 through s+3 all set
    # That's a 5-bit window of 1s starting at bit position s-1
    
    # Build the bitmask as an integer
    # Bit position i corresponds to residue i+1
    mask = 0
    for r in range(1, p):
        if is_residue[r]:
            mask |= (1 << (r - 1))
    
    # For a given s (1-indexed), the 5 residues are s, s+1, s+2, s+3, s+4
    # Their bit positions are s-1, s, s+1, s+2, s+3
    # The 5-bit mask to check is 0b11111 = 31, shifted by (s-1)
    # We need (mask >> (s-1)) & 31 == 31
    
    # Scan s from 1 to 1004 in ascending order to find the minimal s
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if bits s-1 through s+3 are all set
        if (mask >> (s - 1)) & 31 == 31:
            # All five residues s..s+4 are quadratic residues
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
    
    raise RuntimeError('No run of 5 consecutive quadratic residues found')
