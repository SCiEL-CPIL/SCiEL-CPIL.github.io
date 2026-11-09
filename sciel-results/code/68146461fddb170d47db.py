import time

def construct():
    p = 1009
    m = 5
    
    # Precompute quadratic residues
    # We need roots r such that r^2 % p == residue
    # We want the smallest root? The problem says "Each residue is certified by its root".
    # It doesn't specify which root, just that a root exists.
    # However, the return format is {'start':s, 'roots':[r0,...,r4]}.
    # Usually, any valid root is fine. Let's pick the smallest positive root for consistency.
    
    roots_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r not in roots_map:
            roots_map[r] = x
        # If we wanted the smallest root, we'd need to check if x < roots_map[r]
        # But since we iterate x from 1 to p-1, the first one found is the smallest.
        # Wait, x^2 and (p-x)^2 are the same.
        # If x < p-x, i.e., x < p/2, then x is the smaller root.
        # Since we iterate x from 1, the first x that produces r is the smallest root.
        # So the logic above is correct for smallest root.
        
    # We need to find min s such that s, s+1, ..., s+4 are all in roots_map
    # s is in range [1, 1004]
    # s+4 <= 1008, so s <= 1004.
    # Also residues must be non-zero. roots_map keys are non-zero by construction (x in 1..p-1).
    
    deadline = time.time() + 50.0 # Safety margin
    
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            # Fallback: return a known valid solution if we timed out
            # But this should never happen for p=1009
            raise TimeoutError("Search timed out")
            
        # Check if s, s+1, ..., s+4 are all quadratic residues
        valid = True
        for i in range(m):
            val = s + i
            if val not in roots_map:
                valid = False
                break
        
        if valid:
            # Found the minimum s
            res_roots = [roots_map[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
            
    # If no solution found (should not happen)
    raise RuntimeError("No run found")
