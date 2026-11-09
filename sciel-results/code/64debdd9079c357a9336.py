"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    p = 1009
    m = 5  # run length
    
    # Precompute quadratic residues and one root for each
    roots = {}
    bitset = 0
    
    for x in range(1, p):
        v = (x * x) % p
        if v not in roots:
            roots[v] = x
        bitset |= (1 << v)
        
    # Clear bit 0 (zero residue excluded)
    bitset &= ~(1 << 0)
    
    window_mask = (1 << m) - 1
    
    best_s = None
    for s in range(1, p - m + 1):
        if (bitset >> s) & window_mask == window_mask:
            best_s = s
            break
            
        # Optional: deadline check inside loop if p was larger
        # if time.time() > deadline:
        #     break

    if best_s is None:
        raise RuntimeError('No run of length 5 found')
    
    result_roots = []
    for i in range(m):
        val = best_s + i
        r = roots[val]
        result_roots.append(r)
    
    return {'start': best_s, 'roots': result_roots}
# EVOLVE-BLOCK-END
