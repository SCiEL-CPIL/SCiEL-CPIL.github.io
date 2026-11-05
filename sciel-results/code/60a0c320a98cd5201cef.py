import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute non-zero quadratic non-residues
    # is_nr[r] is True if r is a non-zero quadratic non-residue
    is_nr = [False] * (p + 1)
    roots = [0] * (p + 1)
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # r is a residue, so not a non-residue
            # Store root for later
            if roots[r] == 0:
                roots[r] = x
        else:
            # r is 0, excluded
            pass
        
        # Mark non-residues? 
        # Easier: initialize all 1..p-1 as non-residues, then mark residues.
        if time.time() > deadline:
            break

    # Let's stick to the bitmask or a simple list of residues.
    # The prompt specifically asked for "Precompute the list of non-zero quadratic non-residues".
    # Let's do that.
    
    # Reset for the specific strategy requested
    is_nr = [True] * (p + 1)
    is_nr[0] = False # 0 is not a non-zero non-residue in the context of "excluded", but let's be precise.
    # Task: "zero residues ... are excluded". 
    # We want s..s+4 to be QRs.
    # So s..s+4 must NOT be non-residues AND must NOT be 0.
    # Since s>=1 and s<=1004, s+i <= 1008 < 1009, so 0 is not in the range.
    # So we just need to ensure they are not non-residues.
    
    # Initialize is_nr for 1..p-1 as True (assume non-residue)
    for i in range(1, p):
        is_nr[i] = True
        
    roots = [0] * (p + 1)
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_nr[r] = False # r is a residue
            if roots[r] == 0:
                roots[r] = x
        if time.time() > deadline:
            break

    # Scan s from 1 to 1004
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all residues (i.e., not non-residues)
        if not (is_nr[s] or is_nr[s+1] or is_nr[s+2] or is_nr[s+3] or is_nr[s+4]):
            return {'start': s, 'roots': [roots[s], roots[s+1], roots[s+2], roots[s+3], roots[s+4]]}
            
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')
