import time

def construct():
    p = 1009
    
    # Precompute roots for all non-zero quadratic residues
    # roots[r] = x such that x^2 % p == r and 1 <= x <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # We want any root, the first one found is fine.
            # Since x and p-x give the same residue, we might store the smaller one or just the first.
            # The task just requires ri**2 mod 1009 == s+i.
            if r not in roots:
                roots[r] = x
    
    # Precompute a set or boolean array for fast lookup
    # Using a set is O(1) average lookup.
    residue_set = set(roots.keys())
    
    # Search for the minimum s
    # s must satisfy 1 <= s <= 1004 such that s, s+1, s+2, s+3, s+4 are all in residue_set
    # Max value for s+i is s+4. Since s <= 1004, s+4 <= 1008, which is < p.
    # So no wraparound or modulo issues for the values s+i themselves.
    
    for s in range(1, 1005):
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        if (s in residue_set and 
            (s+1) in residue_set and 
            (s+2) in residue_set and 
            (s+3) in residue_set and 
            (s+4) in residue_set):
            
            # Construct the roots list
            r0 = roots[s]
            r1 = roots[s+1]
            r2 = roots[s+2]
            r3 = roots[s+3]
            r4 = roots[s+4]
            
            return {'start': s, 'roots': [r0, r1, r2, r3, r4]}
            
    # If no run is found (should not happen for p=1009, but just in case)
    raise RuntimeError("No valid run found")
