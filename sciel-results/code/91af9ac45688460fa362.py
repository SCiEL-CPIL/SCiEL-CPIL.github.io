def construct():
    p = 1009
    m = 5
    
    # Precompute the smallest root for each non-zero quadratic residue mod p.
    # For each x in [1, p-1], compute r = x^2 % p.
    # root_map[r] = smallest x such that x^2 % p == r.
    root_map = {}
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x
    
    # Find the minimum s in [1, p-m] such that s, s+1, ..., s+m-1 are all in root_map.
    # Direct forward search with O(1) lookup per candidate.
    best_s = None
    for s in range(1, p - m + 1):
        # Check if s, s+1, ..., s+4 are all quadratic residues
        found = True
        for i in range(m):
            if (s + i) not in root_map:
                found = False
                break
        if found:
            best_s = s
            break
    
    if best_s is None:
        raise RuntimeError("No run of 5 consecutive nonzero quadratic residues found")
    
    roots = [root_map[best_s + i] for i in range(m)]
    
    return {'start': best_s, 'roots': roots}
