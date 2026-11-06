import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    p = 1009
    m = 5
    
    # Precompute roots for all non-zero quadratic residues
    # root[r] = x such that x*x % p == r and 1 <= x <= p-1
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x
            
    # Create a set for O(1) membership testing
    residue_set = set(roots.keys())
    
    # Search for minimum s
    # s must be such that s, s+1, ..., s+m-1 are all in residue_set
    # Max s is p - m = 1004
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            break
            
        # Check if s, s+1, ..., s+m-1 are all residues
        valid = True
        res_roots = []
        for i in range(m):
            val = s + i
            if val not in residue_set:
                valid = False
                break
            res_roots.append(roots[val])
        
        if valid:
            return {'start': s, 'roots': res_roots}
            
    # Fallback (should not happen for p=1009, m=5)
    raise RuntimeError('No run found')
