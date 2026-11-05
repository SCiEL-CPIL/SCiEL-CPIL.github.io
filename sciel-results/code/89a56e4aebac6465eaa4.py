import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 90s kill
    
    p = 1009
    m = 5
    
    # Precompute quadratic residues and their roots
    # We store the smallest root x (1 <= x < p) for each residue r
    # If r is not a residue, root[x] remains -1
    roots = [-1] * p
    
    for x in range(1, p):
        r = (x * x) % p
        # Since we iterate x from 1, the first x we find for a residue r is the smallest
        if roots[r] == -1:
            roots[r] = x
            
    # Search for the minimum start s
    # s, s+1, ..., s+4 must all be non-zero quadratic residues
    # Note: s+i must be in range [1, 1008] to be a valid non-zero residue index in our array
    # The problem states 1 <= s <= 1004 and 1 <= r_i <= 1008.
    # If s=1005, s+4=1009=0, which is excluded. So max s is 1004 (s+4=1008).
    
    for s in range(1, 1005):
        if time.time() > deadline:
            break
            
        valid = True
        # Check if s, s+1, s+2, s+3, s+4 are all valid non-zero QRs
        for i in range(m):
            val = s + i
            # val is between 1 and 1008
            if roots[val] == -1:
                valid = False
                break
                
        if valid:
            res_roots = [roots[s + i] for i in range(m)]
            return {'start': s, 'roots': res_roots}
            
    raise RuntimeError('No valid run found within deadline')
# EVOLVE-BLOCK-END
