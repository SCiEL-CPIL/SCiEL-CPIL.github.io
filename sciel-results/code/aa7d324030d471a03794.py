import math
import time

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    # Strategy: Use the chain of powers of 2 as a backbone.
    # This guarantees disjointness with residues r = 2^(k-1) for m = 2^k.
    # Then, try to add other moduli greedily if a valid residue exists.
    
    pairs = []
    # Add powers of 2
    m = 2
    while m <= 128:
        r = m // 2
        pairs.append([r, m])
        m *= 2
        
    # Try to add other moduli
    # Sort remaining moduli by value
    used_moduli = {p[1] for p in pairs}
    candidates = [m for m in range(2, 129) if m not in used_moduli]
    
    for m in candidates:
        # Check if there is a valid residue for m
        valid_r = None
        for r in range(m):
            ok = True
            for r2, m2 in pairs:
                if not _is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                valid_r = r
                break
        if valid_r is not None:
            pairs.append([valid_r, m])
            
    return pairs
