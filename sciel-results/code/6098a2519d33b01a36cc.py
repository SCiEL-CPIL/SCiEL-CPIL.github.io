"""Find the minimum start s for 5 consecutive non-zero quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Step 1: Precompute roots for all non-zero quadratic residues.
    # Map residue -> one root x such that x^2 mod p == residue.
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
    
    # Step 2: Use a sliding window of size 5 over sorted residues.
    # Get sorted list of non-zero quadratic residues.
    residues = sorted(roots.keys())
    
    # We need to find the smallest s such that s, s+1, s+2, s+3, s+4 are all in roots.
    # Use a boolean array for O(1) lookup, then scan s from 1 upward.
    # Since p=1009, this is very fast.
    
    # Build a boolean array: is_residue[i] = True if i is a non-zero QR mod p
    is_residue = [False] * (p + 5)  # extra space to avoid bounds check
    for r in residues:
        is_residue[r] = True
    
    # Scan for the minimum s where s..s+4 are all residues
    # s must be in [1, 1004] since s+4 <= 1008
    for s in range(1, p - 3):  # s+4 <= 1008, so s <= 1004
        if is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]:
            return {'start': s, 'roots': [roots[s+i] for i in range(5)]}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
