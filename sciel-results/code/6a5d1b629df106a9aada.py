"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute is_res and roots
    is_res = [0] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        is_res[r] = 1
        if roots[r] == 0:
            roots[r] = x
            
    # Compute prefix sums of is_res
    # prefix[i] = sum(is_res[0] ... is_res[i-1])
    # So sum(is_res[s] ... is_res[s+4]) = prefix[s+5] - prefix[s]
    prefix = [0] * (p + 5)
    for i in range(p):
        prefix[i + 1] = prefix[i] + is_res[i]
        
    # Find the smallest s such that s, s+1, ..., s+4 are all QR
    # s+4 <= 1008 => s <= 1004, and s >= 1
    for s in range(1, 1005):
        # Check if all 5 consecutive values are QR
        if prefix[s + 5] - prefix[s] == 5:
            # Retrieve precomputed roots
            res_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': res_roots}
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END
