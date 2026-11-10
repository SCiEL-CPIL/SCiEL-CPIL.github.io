import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 60.0 # 60s limit, use 50s to be safe? "KILLS after 90s". "Safety margin".
    
    # Build residue mask and roots
    # roots[i] stores smallest x in [1, p-1] such that x^2 % p == i
    roots = [0] * p
    mask = 0
    
    for x in range(1, p):
        if time.time() > deadline - 10: # Check time occasionally
            break
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if roots[r] == 0 or x < roots[r]:
                roots[r] = x

    # Find s: smallest index where 5 consecutive bits are set
    # common has bit k set iff k, k+1, k+2, k+3, k+4 are all set in mask
    common = mask
    common &= mask >> 1
    common &= mask >> 2
    common &= mask >> 3
    common &= mask >> 4
    
    if common == 0:
        # Fallback: linear search (should not happen for p=1009, m=5)
        # But just in case
        for s in range(1, 1005):
            if all((mask >> (s + i)) & 1 for i in range(5)):
                break
        else:
            raise RuntimeError("No run of 5 quadratic residues found")
    else:
        low_bit = common & -common
        s = low_bit.bit_length() - 1
        
        # Ensure s is in valid range
        if s < 1 or s > 1004:
            # Find valid s
            for s_candidate in range(1, 1005):
                if (common >> s_candidate) & 1:
                    s = s_candidate
                    break
            else:
                raise RuntimeError("No valid s in range")

    res_roots = [roots[s + i] for i in range(5)]
    
    return {'start': s, 'roots': res_roots}
