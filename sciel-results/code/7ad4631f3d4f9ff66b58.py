"""Find the minimum start s for 5 consecutive non-zero quadratic residues mod 1009."""
# EVOLVE-BLOCK-START
import time

_P = 1009

def construct():
    p = _P
    t0 = time.time()
    deadline = t0 + 50.0  # 50s safety margin
    
    # Precompute: for each non-zero residue, store the smallest root
    root_map = {}
    # Bitmask for fast residue checking
    residue_mask = 0
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if r not in root_map:
            root_map[r] = x
            residue_mask |= (1 << r)
        
        # Time check (unlikely to trigger for p=1009)
        if time.time() > deadline:
            break
    
    # Search ascending to minimize s
    # s must be in [1, 1004] so that s+4 <= 1008
    # We need s, s+1, s+2, s+3, s+4 to all be non-zero quadratic residues
    # Pattern for 5 consecutive bits: 0b11111 = 31
    pattern = 31
    
    max_s = p - 5  # 1004
    
    for s in range(1, max_s + 1):
        # Check if bits s, s+1, s+2, s+3, s+4 are all 1 in residue_mask
        if (residue_mask >> s) & pattern == pattern:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
        
        if time.time() > deadline:
            break
    
    # Safety: should not happen for p=1009, but avoid hanging
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
