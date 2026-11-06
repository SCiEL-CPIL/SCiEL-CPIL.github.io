import time

_P = 1009

def construct():
    p = _P
    t0 = time.time()
    deadline = t0 + 50.0
    
    # Precompute roots and residue mask
    root_map = {}
    residue_mask = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x
            residue_mask |= (1 << r)
        
        if time.time() > deadline:
            break

    # Search for minimum s
    # We need s, s+1, s+2, s+3, s+4 to be residues.
    # s is in [1, 1004]
    pattern = 31 # 0b11111
    max_s = p - 5
    
    for s in range(1, max_s + 1):
        # Check if bits s through s+4 are set in residue_mask
        # Shift right by s, then mask with pattern
        if ((residue_mask >> s) & pattern) == pattern:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
        
        if time.time() > deadline:
            break

    raise RuntimeError('No valid run found')
