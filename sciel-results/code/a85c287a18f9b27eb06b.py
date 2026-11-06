"""Rewritten program to find the minimum start s for 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    
    # Precompute roots for all non-zero quadratic residues.
    # Map residue -> root (any valid root works).
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We only need one root per residue; first found is fine.
            if r not in roots:
                roots[r] = x
    
    # Build a boolean array indicating which values 1..p-1 are quadratic residues.
    # is_residue[i] is True if i is a non-zero quadratic residue mod p.
    is_residue = [False] * p
    for r in roots:
        is_residue[r] = True
    
    # Sliding window of size 5 over values 1..p-5 (since s+i must be <= p-1, i.e., <= 1008)
    # s ranges from 1 to p-5 = 1004.
    # We want the smallest s such that is_residue[s], is_residue[s+1], ..., is_residue[s+4] are all True.
    
    best_s = None
    for s in range(1, p - 4):  # s from 1 to 1004
        if is_residue[s] and is_residue[s+1] and is_residue[s+2] and is_residue[s+3] and is_residue[s+4]:
            best_s = s
            break
    
    if best_s is None:
        raise RuntimeError("No run of 5 consecutive quadratic residues found")
    
    result_roots = [roots[best_s + i] for i in range(5)]
    
    return {'start': best_s, 'roots': result_roots}
