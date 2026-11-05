import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    p = 1009
    
    # Precompute quadratic residues and a representative root for each
    # is_residue[i] is True if i is a non-zero quadratic residue mod p
    # root_map[i] stores a root r such that r*r % p == i, for non-zero residues
    is_residue = [False] * p
    root_map = [0] * p
    
    # Also build the bitmask
    # Bit i is 1 if i is a non-zero QR
    mask = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_residue[r] = True
            if root_map[r] == 0:
                root_map[r] = x
            mask |= (1 << r)
    
    # We need s, s+1, s+2, s+3, s+4 to all be non-zero quadratic residues.
    # s must be between 1 and 1004 (so s+4 <= 1008).
    # We minimize s, so we iterate from 1 upwards.
    
    for s in range(1, 1005):
        if time.time() > deadline:
            break
            
        # Check if s, s+1, s+2, s+3, s+4 are all quadratic residues
        # Using bitmask: shift right by s, check if the lowest 5 bits are all 1
        # 0x1F is 0b11111
        if ((mask >> s) & 0x1F) == 0x1F:
            return {
                'start': s,
                'roots': [
                    root_map[s],
                    root_map[s+1],
                    root_map[s+2],
                    root_map[s+3],
                    root_map[s+4]
                ]
            }
            
    # Fallback: should not happen for p=1009, but provide a safe return
    # If we get here, something is wrong; raise an error
    raise RuntimeError("No valid run of 5 consecutive non-zero QRs found modulo 1009")
# EVOLVE-BLOCK-END
